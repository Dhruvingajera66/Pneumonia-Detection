# 🩺 Pneumonia Classification using Deep Learning

🚀 This project uses **Deep Learning** to classify chest X-ray images into **Normal** or **Pneumonia** using a trained model. A **Streamlit-based web application** is developed for real-time predictions.

---

## **1️⃣ Setup for Python**
- 🛠 **Install Python**: Follow [setup instructions](https://www.python.org/downloads/).
- 📦 **Install dependencies**:
  ```bash
  pip3 install -r training/requirements.txt


## 2️⃣ Training the Model

### 📥 Download the dataset:
- Get chest X-ray images from Kaggle or another medical dataset source.
- Keep only folders for Normal and Pneumonia cases.

### 🚀 Run Jupyter Notebook or .py file:
```bash
jupyter notebook/ .py file
```

### 📂 Open the training script:
```bash
training/Model_1.py
```

### ▶️ Run all cells to train the model.

### 💾 Save the trained model in the models/ directory:
```bash
Final_models/
└── Version_1.h5
```

---

## 3️⃣ Running the Streamlit Web App

### 📦 Install required packages:
```bash
pip3 install -r frontend/requirements.txt
```

### 📂 Navigate to the frontend directory:
```bash
cd frontend
```

### 🚀 Run the Streamlit app:
```bash
streamlit run app.py
```

### 🖼 Upload a chest X-ray image and get real-time pneumonia detection results!

---

## 4️⃣ Web App Features
- 🖼 Upload an X-ray image via the Streamlit interface.
- 🔍 Automatically preprocess and classify the image using the trained model.
- 📊 Real-time visualization of results.

## 5️⃣ Conclusion
✅ This project provides a simple but effective AI-based Pneumonia Detection system using:
- Deep learning model for medical image classification
- Streamlit web application for real-time predictions

---

## 📌 Project Structure
```bash
Pneumonia-Detection/
│── Final_models/                      # Saved model files
│   ├── Version_1.h5        # Trained deep learning model
│── training/
│   ├── model_1.py # Jupyter Notebook for training
│   ├── requirements.txt          # Dependencies for training
│── frontend/
│   ├── app.py                    # Streamlit web app
│   ├── requirements.txt          # Dependencies for Streamlit app
│── README.md                     # Project documentation
```

---

## 💡 Future Improvements
- 🔹 Improve dataset diversity and size
- 🔹 Enhance the model architecture for better accuracy
- 🔹 Deploy the model as an API using FastAPI or Flask

---

## 🌟 Support & Contribution
⭐ If you find this project helpful, star the repository and feel free to contribute!
🔧 For issues or suggestions, open an issue or pull request.

---

## ✅ Conclusion
✔️ This project provides a complete deep learning pipeline for pneumonia detection, covering:
- Model Training (CNN-based architecture)
- Web App for Predictions (Streamlit-based UI)
- User-friendly & Easy-to-use Interface

---


