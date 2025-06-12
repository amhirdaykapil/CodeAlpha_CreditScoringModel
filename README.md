# 💳 Credit Scoring Prediction Model — CodeAlpha Internship

This project predicts whether a person is likely to default on a loan using real-world financial dataset.

---

## 🚀 Features

- Random Forest model using scikit-learn
- Predicts loan default (0 = safe, 1 = risky)
- Random user prediction from dataset
- MySQL-style terminal output formatting
- Evaluation metrics : Accuracy, Recall, ROC-AUC
- Feature Importance Graph

---

## 📂 Project Structure

CodeAlpha_CreditScoringModel/<br>
├── cs-training.csv<br>
├── credit_scoring_individual_predict.py<br>
├── requirements.txt<br>
└── README.md<br>

---

## ⚙️ Installation

Install the required Python libraries using: _pip install -r requirements.txt_

---

## 📂 Dataset Used `cs-training.csv` by [XiaoChan](https://www.kaggle.com/carolline89) from [Kaggle](https://www.kaggle.com/datasets/carolline89/cs-training/data).

---

## ⚙️ Technologies Used

- Python 🐍
- pandas, matplotlib, seaborn
- scikit-learn (Random Forest Classifier)

---

## 🖼️ Screenshot (Sample Output)

Below is a terminal output screenshot of a **random user** from the dataset who was predicted to be **safe** by the model, i.e., not likely to default.

![Prediction Output Screenshot](screenshot.png)

Below is a terminal output screenshot of a **random user** from the dataset who was predicted as a **high-risk defaulter** by the model.

![Prediction Output Screenshot](screenshot2.png)

The Random Forest model analyzes the user's financial profile and determined that whether the individual is **HIGH RISK DEFAULTER** or **SAFE**.

---

## ▶️ How to Run

Run the main Python file using: _python credit_scoring_model.py_

The script will:
- Train a model using Random Forest
- Show evaluation (confusion matrix, accuracy, etc.)
- Randomly select one user from dataset
- Predict their loan default probability

---

## ✍️ Author

👤 Hirday Kapil<br>
🎓 CodeAlpha Machine Learning Intern<br>
📅 June 2025<br>
