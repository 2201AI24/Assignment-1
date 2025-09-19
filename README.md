# 🧬 Breast Cancer Classification with SVM  

This project is an implementation of **Support Vector Machine (SVM) classifiers** to predict whether a breast tumor is **benign** or **malignant**, using the **Breast Cancer Wisconsin dataset**.  

---

## 📌 Overview  

The notebook explores the dataset, applies preprocessing, trains multiple **SVM kernels (Linear, Polynomial, RBF, Sigmoid)**, tunes hyperparameters, and evaluates model performance.  
The goal is to compare kernels and determine the best-performing classifier for medical diagnosis tasks.  

---

## 🚀 Features  

* Load and analyze the **Breast Cancer Wisconsin dataset** (from scikit-learn)  
* Perform **EDA** and visualize feature distributions  
* Apply **data preprocessing** (scaling, train-test split)  
* Train **SVM classifiers** with Linear, Poly, RBF, and Sigmoid kernels  
* Perform **hyperparameter tuning** using GridSearchCV  
* Evaluate models using **accuracy, precision, recall, F1-score, confusion matrix**  
* Visualize **decision boundaries and performance metrics**  

---

## 🛠️ Technologies Used  

* Python  
* pandas & NumPy  
* matplotlib & seaborn  
* scikit-learn (SVM, preprocessing, model evaluation)  
* Google Colab  

---

## 🧠 Model Details  

The **SVM classifier** is tested with four kernels:  

* **Linear** – baseline performance  
* **Polynomial** – nonlinear separation  
* **RBF (Radial Basis Function)** – flexible and widely used kernel  
* **Sigmoid** – alternative nonlinear mapping  

### Preprocessing Steps  

* **Data split:** 80% training, 20% testing  
* **Scaling:** StandardScaler to normalize features  
* **Tuning:** GridSearchCV for parameters like `C`, `gamma`, and `degree`  

---

## 📊 Results  

| Kernel     | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|--------|----------|
| Linear     | XX%      | XX%       | XX%    | XX%      |
| Polynomial | XX%      | XX%       | XX%    | XX%      |
| RBF        | XX%      | XX%       | XX%    | XX%      |
| Sigmoid    | XX%      | XX%       | XX%    | XX%      |  

👉 The **best kernel** was identified based on F1-score and overall balance of precision/recall.  

---

## 📷 Sample Output  

*(Example: confusion matrix, decision boundary, or accuracy plot can be shown here)*  

---

## 📦 How to Run  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/Assignment-1.git
   cd Assignment-1

2.Open the notebook in Jupyter/Colab:
  jupyter notebook Assignment1.ipynb

3.Run all cells to reproduce the results.

---

## 📁 Project Structure

```
├── app.py
├── model.ipynb 
├── xgb_churn_model.pkl
├── feature_columns.pkl
├── label_encoders.pkl
├── .env
├── requirements.txt
├──README.md
└──WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## 👨‍💻 Author

M. Umesh Chandra<br>
BTech Artificial Intelligence and Data Science (Batch 2022)<br> 
Project: Telecom Churn Prediction + LLM Explanation

---

## 📄 License

This project is for educational use only.
