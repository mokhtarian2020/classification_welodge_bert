✉️ Smart Email Classifier for Italian Business
Welcome to your intelligent email classification system — built to streamline how Italian companies manage and route their emails. Powered by state-of-the-art AI (BERT for Italian), this solution reads incoming emails and instantly sorts them into the right business department.

No more misrouted emails. No more delays.

🧠 What It Does
Understands emails written in Italian, including subjects, bodies, and even attachments.

Assigns each email to the correct business category like HR, Legal, Accounting, Customer Service, and more.

Learns from feedback — when a user corrects a label, the system remembers and improves.

Retrains itself every month automatically with fresh feedback to stay sharp.

Runs as an API you can integrate into any app, web tool, or internal system.

Supports GPU acceleration and is ready to scale using Docker.

🔧 Built With Modern Technologies
Transformers + PyTorch for training and running a custom Italian BERT model.

FastAPI for fast and modern RESTful API endpoints.

APScheduler for automated monthly retraining.

Faker for generating realistic Italian email examples.

Pandas & Scikit-learn for data processing and metrics.

Docker & NVIDIA runtime for robust, GPU-powered deployment.

🗂 Categories the AI Can Detect
The model can accurately classify emails into key corporate areas such as:

Legal

Accounting

Invoicing

Company Vehicles

Human Resources

Procurement

Customer Service

IT & Technical Services

These are customizable and easily expandable based on your organization’s needs.

🧾 How the Input Works
The system expects a JSON-formatted email with:

A subject line (soggetto)

The main email body (corpo)

Optional attachments with extracted text (allegati)

It processes the entire content, even if long, by breaking it into chunks the model can handle — ensuring accurate classification every time.

🔁 Always Getting Smarter
Every time the model mislabels something and the user provides the correct label, it logs that feedback. Then, on the 1st of every month, it:

Gathers all feedback

Retrains the model

Replaces the old one — automatically

No manual updates. No forgotten training runs.

🐳 Ready for Production
Everything is containerized using Docker and ready to go with one command. It supports:

GPU inference for fast response times

Live API endpoints for classification and feedback

Persistent feedback logging and retraining

Perfect for enterprise environments or integration into existing CRMs and helpdesk systems.

🚀 Summary
This is not just a classifier — it's a smart assistant that evolves with your workflow. Whether you're managing hundreds or thousands of emails a week, this system helps route them to the right place, right away.
