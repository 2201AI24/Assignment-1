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
| Kernel  | Test Accuracy |
| ------- | ------------- |
| Linear  | 0.9649        |
| Poly    | 0.9474        |
| RBF     | 0.9649        |
| Sigmoid | 0.9123        |

* Best Kernel: linear (or rbf)

*  Test Accuracy: 0.9649

*  SVM with linear/RBF kernel can accurately classify breast cancer instances with minimal misclassifications.

| Class           | Precision | Recall | F1-Score | Support |
| --------------- | --------- | ------ | -------- | ------- |
| 0 (Benign)      | 0.99      | 0.97   | 0.98     | 70      |
| 1 (Malignant)   | 0.96      | 0.98   | 0.97     | 44      |
| **Avg / Total** | 0.97      | 0.97   | 0.97     | 114     |

[[72  0]
 [ 3 39]]
 
* Interpretation:

  *  TN: 68, FP: 2, FN: 1, TP: 43

*  Only 3 misclassifications → strong performance.

*  Best Parameters: {'C': 1, 'kernel': 'linear', 'gamma': 'scale', 'degree': 3}

*  Cross-Validation Accuracy: 0.9649

*  Test Accuracy with Best Model: 0.9649

---

## 📷 Sample Output  

<img width="613" height="451" alt="image" src="https://github.com/user-attachments/assets/25e2e06a-e7dc-450c-8f01-12a4b7bf918e" />
<img width="365" height="316" alt="image" src="https://github.com/user-attachments/assets/034f1305-4117-4ef1-bede-cd85cda17801" />
<img width="625" height="470" alt="image" src="https://github.com/user-attachments/assets/8aaa9c5e-a792-48fe-9361-2165f6b5193d" />

*(Example: confusion matrix, decision boundary, or accuracy plot can be shown here)*  

---

## 📦 How to Run  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/Assignment-1.git
   cd Assignment-1

2.Open the notebook in Jupyter/Colab:
  ```
  jupyter notebook Assignment1.ipynb
  ```
3.Run all cells to reproduce the results.

---

## 📁 Project Structure

```
├── Assignment1.ipynb   # Colab notebook with full code
├── Assignment1.pdf     # Exported PDF report
├── README.md           # Project documentation
└──requirements.txt
```

---

## 👨‍💻 Author

M. Umesh Chandra<br>
BTech Artificial Intelligence and Data Science (Batch 2022)<br> 
Roll No: 2201AI24
Course: Advance Pattern Recognition – Assignment 1

---

## 📄 License

This project is for educational use only.
