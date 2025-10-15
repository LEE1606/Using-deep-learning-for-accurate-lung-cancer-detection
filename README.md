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


<img width="486" height="214" alt="image" src="https://github.com/user-attachments/assets/20e5e2d5-23db-4fc0-89f5-58298d063eea" />
<img width="565" height="139" alt="image" src="https://github.com/user-attachments/assets/f1e0c15b-4e25-4c07-a07f-9716f2d5c247" />
<img width="446" height="239" alt="image" src="https://github.com/user-attachments/assets/988c9906-61c1-4758-b903-adae5c968cd4" />
<img width="425" height="300" alt="image" src="https://github.com/user-attachments/assets/e91b28b6-173a-4dc9-a8e6-65f5e68e5c0a" />
<img width="464" height="161" alt="image" src="https://github.com/user-attachments/assets/cfa36aff-5184-4175-8da3-058ab910b941" />

