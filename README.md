
![Project Banner](image.png)

# 📌 **Urdu Deepfake Audio Detection — README**

## 🧠 **Project Overview**

This project focuses on detecting **deepfake vs. real Urdu speech**, using machine-learning models trained on audio features. With the rise of manipulated audio in politics, media, and social networks, this system helps identify fake voice recordings with high accuracy.

You built a **complete ML pipeline** that includes:

- Dataset preprocessing  
- Feature extraction (MFCCs)  
- Model training (ML + DL)  
- Model comparison & visualization  
- Gradio-based interactive interface  

---

## 🎯 **Objectives**

- Accurately classify **real vs. fake Urdu audio**  
- Build a **scalable workflow** for future dataset expansion  
- Provide a **user-friendly detection interface** using Gradio  
- Compare multiple algorithms to find the optimal model  

---

## 📂 **Project Structure**

```

Urdu_Deepfake_Audio_Detection/
│
├── dataset/
│   ├── real/
│   ├── fake/
│
├── preprocessed/
│   ├── X.npy        # MFCC feature matrix
│   ├── y.npy        # Labels (0 = real, 1 = fake)
│
├── models/
│   ├── logistic_regression.pkl
│   ├── svm_model.pkl
│   ├── random_forest.pkl
│
├── Urdu_Deepfake_Audio_Detection_.ipynb
├── model_comparison_bar_chart.png
└── README.md

```

---

## ⚙️ **How the System Works**

### **1️⃣ Dataset Loading**
Files from `real/` and `fake/` folders are scanned and labeled.

### **2️⃣ Preprocessing**
- Audio converted to mono  
- Resampled to 16kHz  
- 20 MFCCs extracted  
- Features padded or truncated to fixed size  

### **3️⃣ Model Training**
Models used:
- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  

Labels:
- **0 → Real Audio**  
- **1 → Deepfake Audio**

### **4️⃣ Evaluation**
A bar chart compares:
- Accuracy  
- Precision  
- Recall  
- F1-score  

Saved as:
```

model_comparison_bar_chart.png

````

### **5️⃣ Gradio App**
Upload any `.wav` file → app predicts:
- Real / Fake  
- Confidence score  

Deployable via:
- Local server  
- HuggingFace  
- Streamlit  

---

## 🛠️ **Technologies Used**

| Component | Libraries |
|----------|-----------|
| Audio Processing | Librosa |
| Machine Learning | Scikit-Learn |
| Visualization | Matplotlib |
| Interface | Gradio |
| Data Handling | NumPy, Pandas |
| Notebook | Jupyter |

---

## 📊 **Model Performance Overview**

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | Good baseline | Fast, simple |
| SVM | High accuracy | Very effective for MFCCs |
| Random Forest | Competitive | Captures nonlinear patterns |

*(Metrics vary based on dataset)*

---

## 🚀 **How to Run**

### **1. Install Dependencies**
```bash
pip install numpy librosa scikit-learn gradio matplotlib
````

### **2. Preprocess Dataset**

```python
process_dataset("dataset/")
```

### **3. Train Models**

```bash
python train_models.py
```

### **4. Launch Gradio App**

```bash
python app.py
```

---

## 🎤 **Gradio App Features**

* Upload Urdu audio
* Real-time deepfake classification
* Confidence score output
* Clean UI suitable for demos

---

## 🧩 **Future Improvements**

* Add CNN/LSTM deep-learning models
* Use larger datasets
* Noise-robust feature extraction
* Deploy API + Mobile App

---

## 🏅 **Use Cases**

* Journalism fact-checking
* Social media misinformation detection
* Law enforcement
* Political authentication
* Academic/educational use

---
