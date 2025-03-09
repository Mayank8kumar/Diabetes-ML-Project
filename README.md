# **Diabetes Prediction using KNN**  

## **📌 Project Overview**  
This machine learning project predicts diabetes using the **K-Nearest Neighbors (KNN) algorithm**. The dataset used contains medical diagnostic measurements, and preprocessing techniques have been applied to improve model performance.  

---

## **📂 Project Structure**  
📂 Diabetes-ML-Project │-- 📄 README.md # Project documentation │-- 📂 data
│ ├── diabetes.csv # Dataset used for training │-- 📂 notebooks
│ ├── diabetes.ipynb # Data exploration & preprocessing │ ├── Machine learning project.ipynb # Model training (KNN) │-- 📄 .gitignore # Ignore unnecessary files


---

## **📊 Dataset Information**  
- The dataset contains **medical diagnostic features** to predict whether a patient has diabetes.  
- It includes **independent variables** like glucose level, blood pressure, insulin levels, BMI, and more.  
- The **target variable** is `"Outcome"`:  
  - `1` → **Diabetic**  
  - `0` → **Non-Diabetic**  

---

## **⚙️ Preprocessing Steps**  
To improve model accuracy, the following preprocessing steps were applied:  

✅ Handling missing values (if any)  
✅ Feature scaling for better KNN performance  
✅ Train-test split for model evaluation  

---

## **🤖 Machine Learning Model (KNN)**  
- The **K-Nearest Neighbors (KNN)** algorithm was used for classification.  
- The model finds the **K closest data points** and predicts the most common class.  
- Hyperparameter tuning was applied to select the best **K value**.  

---

## **📈 Model Evaluation**  
The model was evaluated using:  

- **Accuracy Score** → Measures overall correctness  
- **Confusion Matrix** → Shows true positives/negatives  
- **Precision & Recall** → Evaluates classification performance  

---

## **🚀 How to Run the Project**  

### **1️⃣ Clone the repository**  
```bash
git clone https://github.com/Mayank8kumar/Diabetes-ML-Project.git
cd Diabetes-ML-Project

### **2️⃣ Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn

### **3️⃣ Run Jupyter Notebook**
```bash

jupyter notebook
Open notebooks/diabetes.ipynb for data analysis
Open notebooks/Machine learning project.ipynb to train the KNN model

### **📜 License**
This project is open-source and free to use for educational purposes.

---

✅ **Now you can directly copy-paste this into `README.md`!** 🚀 Let me know if you need any modifications. 😊
