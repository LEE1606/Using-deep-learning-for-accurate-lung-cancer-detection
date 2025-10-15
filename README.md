🫁 Using Machine Learning for Accurate Lung Cancer Detection

This project presents a CT-based lung cancer detection system that combines deep learning classification and unsupervised segmentation into a lightweight and explainable diagnostic workflow. The system integrates EfficientNetB1 for cancer classification and Fuzzy C-Means (FRFCM) segmentation for tumor region localization, wrapped within an interactive Streamlit dashboard and supported by a SQLite database for case management and result archiving.
🚀 Key Features

🧠 EfficientNetB1 Classifier
Detects whether a CT scan indicates lung cancer (benign or malignant).

🔍 FCM Unsupervised Segmentation
Automatically identifies suspicious lung regions using fuzzy clustering and morphological post-processing (small object removal, hole filling, brightness filtering).

💾 SQLite Case Management System
Stores patient metadata, predictions, segmentation results, and radiologist review notes.

📄 AIM XML + PDF Report Generator
Exports structured diagnostic reports following medical imaging standards.

🌐 Streamlit Dashboard
User-friendly web interface to upload CT scans, view model results, visualize masks, and manage patient records.
