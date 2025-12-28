# 🧠 Brain Tumor MRI Image Classification (Deep Learning + Streamlit)

This project is a **deep learning–based medical imaging application** that classifies **brain MRI images** into different tumor categories using **Custom CNN** and **Transfer Learning models**.  
A **Streamlit web app** is provided for real-time tumor prediction from uploaded MRI images.

---

## 🎯 Problem Statement

To build an AI-powered system that can automatically classify brain MRI images into tumor types such as:

- Glioma
- Meningioma
- Pituitary
- No Tumor

This helps radiologists and doctors with **faster diagnosis**, **early detection**, and **decision support**.

---

## 🏥 Real-Time Business Use Cases

- **AI-Assisted Medical Diagnosis**
- **Early Detection & Patient Triage**
- **Clinical Research & Trials**
- **Second-Opinion AI for Telemedicine**

---

## 🚀 Features

- MRI image classification using Deep Learning
- Custom CNN model from scratch
- Transfer Learning with MobileNet
- Image preprocessing & normalization
- Model evaluation & confidence score
- Streamlit-based interactive web app
- Google Drive–based model loading (for deployment)
- Clean & simple UI for doctors and users

---

## 🧾 How to Run this Project

- git clone https://github.com/keshavjatt/Brain-Tumor.git 
- cd Brain-Tumor
- pip install -r requirements.txt
- python train_custom_cnn.py
- python train_transfer.py
- python evaluate.py
- python -m streamlit run app.py