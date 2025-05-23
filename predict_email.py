import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import math
from collections import Counter, defaultdict

# Load model and labels
MODEL_PATH = "outputs/model"
LABELS = pd.read_csv("outputs/label_mapping.csv").squeeze().tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and move to GPU if available
model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model.eval()

# Configuration
MAX_TOKENS = 512  # Bert limit
CHUNK_SIZE = 450  # Slightly smaller than limit to avoid errors

def json_to_bert_input(email_json):
    soggetto = email_json.get("soggetto", "").strip()
    corpo = email_json.get("corpo", "").strip()
    allegati_list = email_json.get("allegati", [])

    # Join texts from attachments (if any)
    allegati_text = " ".join(
        att.get("testo", "").strip()
        for att in allegati_list if isinstance(att, dict)
    ).strip()

    # Always return all parts, including [SEP] separators
    return f"{soggetto} [SEP] {corpo} [SEP] {allegati_text}"

def split_into_chunks(text, chunk_size=CHUNK_SIZE):
    tokens = tokenizer.tokenize(text)
    num_chunks = math.ceil(len(tokens) / chunk_size)

    chunks = []
    for i in range(num_chunks):
        chunk_tokens = tokens[i*chunk_size : (i+1)*chunk_size]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def predict(email_json):
    full_text = json_to_bert_input(email_json)
    tokens = tokenizer.tokenize(full_text)

    if len(tokens) <= MAX_TOKENS:
        # Normal short email
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        return LABELS[pred_id]

    else:
        # Long email: chunk it
        chunks = split_into_chunks(full_text)
        preds = []
        probs_sum = defaultdict(float)

        for chunk in chunks:
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                padding="max_length"
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            preds.append(pred_id)

            # Accumulate probability for predicted label
            probs_sum[pred_id] += probs[0][pred_id].item()

        counts = Counter(preds)
        most_common = counts.most_common()

        if len(chunks) == 2:
            # Special case: exactly 2 chunks
            # Pick the label with higher probability
            first_label, second_label = preds[0], preds[1]
            first_prob = probs_sum[first_label]
            second_prob = probs_sum[second_label]
            if first_prob >= second_prob:
                return LABELS[first_label]
            else:
                return LABELS[second_label]

        else:
            # >2 chunks
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                # Tie detected
                tied_labels = [most_common[0][0], most_common[1][0]]
                best_label = max(tied_labels, key=lambda x: probs_sum[x])
                return LABELS[best_label]
            else:
                # Normal majority
                return LABELS[most_common[0][0]]
