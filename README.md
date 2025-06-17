# 📊 ML Model Evaluator

**ML Model Evaluator** is a web-based Flask application that allows users to upload a dataset (CSV), select features and target columns, choose machine learning models, and view evaluation metrics — all from a clean, interactive interface.

---

## 🔧 Features

- 📁 Upload CSV datasets directly
- 📌 Dynamically select:
  - Target column (Y)
  - Feature columns (X)
  - One or more ML models to run
- 🧠 Supports both **classification** and **regression**
- 📉 View model performance metrics like:
  - Accuracy (for classification)
  - R² score (for regression)
- ✅ Automatically identifies problem type based on the target column
- 🌐 Built with Python, Flask, scikit-learn, and XGBoost

---

## 📦 Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, xgboost
- **Frontend**: HTML, CSS, JavaScript
- **Environment**: Python 3.8+

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ml-model-evaluator.git
cd ml-model-evaluator
