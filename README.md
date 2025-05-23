AI-Powered Email Classification System
This project is an intelligent email classification system designed for Italian business environments. It classifies incoming emails into predefined categories using a fine-tuned BERT model. The system also includes functionality for collecting user feedback, performing monthly automatic retraining, and deploying the model as a Dockerized API service.

Features
Automatically classifies Italian emails into one of several business-related categories based on subject, body, and attachments.

Fine-tuned transformer model (dbmdz/bert-base-italian-xxl-cased) using synthetic but realistic email data.

Collects real user feedback when classification is incorrect and uses it to continuously improve the model.

Retrains the model every month automatically using newly collected feedback data.

Exposes a RESTful API with FastAPI to handle real-time predictions and feedback.

Fully containerized using Docker with GPU support for scalable production deployment.

Technologies Used
HuggingFace Transformers and Datasets for model training and tokenization.

PyTorch and scikit-learn for deep learning and evaluation.

Faker for generating synthetic email content in Italian.

Pandas for data handling and CSV operations.

FastAPI and Pydantic for building and validating REST API services.

APScheduler for scheduling automatic monthly retraining jobs.

Docker and docker-compose with NVIDIA GPU support for deployment.

Project Capabilities
Simulates realistic Italian business emails from categories such as legal, accounting, invoicing, company vehicles, HR, procurement, customer relations, and IT services.

Parses structured JSON inputs containing subject, body, and attachments.

Uses BERT to classify emails and automatically handles long texts by breaking them into chunks.

Allows users to submit feedback when a classification is incorrect, logging the correct label.

Uses this feedback in retraining, replacing the old model with a new one without manual intervention.

Includes a scheduler that triggers retraining automatically on the 1st of each month.

The system is easily extensible to add new categories or input formats.

Deployment and Usage
The system can be trained and tested locally using Python scripts or deployed in production using Docker. The REST API includes endpoints for predicting email labels and submitting feedback. The model and feedback logs are automatically updated after each retraining cycle. Everything is packaged into a container to ensure reproducibility and GPU compatibility.

Input Format
The API accepts structured email input with subject, body, and optional attachments. The attachments are parsed to extract their textual content and included in the classification process. Input must follow a specific JSON format to ensure compatibility with the model.

Output
The API returns the predicted label and the complete input text used by the model. If the user provides the correct label and it differs from the prediction, the system logs the feedback and stores it for future model improvement.

API Availability
Once deployed, the API is accessible on the specified port (default is 8001) and can be tested using tools such as Postman or integrated into external applications.
