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

<img width="771" height="331" alt="image" src="https://github.com/user-attachments/assets/b7098f98-1417-4ace-9af7-eb30e1d330c2" />
<img width="940" height="229" alt="image" src="https://github.com/user-attachments/assets/591c3349-ea36-4bf1-ae46-fd0c9edbfda6" />
<img width="733" height="379" alt="image" src="https://github.com/user-attachments/assets/673022e4-3a14-42cc-8488-86a45b003663" />
<img width="759" height="345" alt="image" src="https://github.com/user-attachments/assets/b929247b-f1b5-4264-b6b5-195787d5cb50" />
<img width="742" height="488" alt="image" src="https://github.com/user-attachments/assets/96eadf42-4747-4867-be61-721d9dfa0597" />
<img width="793" height="282" alt="image" src="https://github.com/user-attachments/assets/0ca0b440-071d-4c2d-a17b-20563c6742ca" />
<img width="735" height="398" alt="image" src="https://github.com/user-attachments/assets/245ba708-b2b2-41a1-ab2f-2467adaebbfb" />




