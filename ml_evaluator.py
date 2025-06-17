# ml_evaluator.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import warnings

warnings.filterwarnings("ignore")

def is_classification_task(y):
    if isinstance(y, np.ndarray):
        return y.dtype == 'object' or len(np.unique(y)) < 20
    return y.dtype == 'object' or y.nunique() < 20

def evaluate_models(X, y, classification=True, selected_models=None):
    if len(X) < 4:
        X_train, X_val, y_train, y_val = X, X, y, y
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
        X_val, _, y_val, _ = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    X_train = pd.get_dummies(X_train)
    X_val = pd.get_dummies(X_val)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    if classification:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree Classifier": DecisionTreeClassifier(),
            "Random Forest Classifier": RandomForestClassifier(),
            "Gradient Boosting Classifier": GradientBoostingClassifier(),
            "AdaBoost Classifier": AdaBoostClassifier(),
            "KNN Classifier": KNeighborsClassifier(),
            "SVC": SVC(),
            "Naive Bayes": GaussianNB(),
            "MLP Classifier": MLPClassifier(max_iter=1000),
            "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "ElasticNet": ElasticNet(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor(),
            "KNN Regressor": KNeighborsRegressor(),
            "SVR": SVR(),
            "MLP Regressor": MLPRegressor(max_iter=1000),
            "XGBoost Regressor": XGBRegressor()
        }

    if selected_models:
        models = {k: v for k, v in models.items() if k in selected_models}

    results = []

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            result = {"Model": name}

            if classification:
                result["Accuracy"] = round(accuracy_score(y_val, y_pred), 4)
                result["R²"] = round(r2_score(y_val, y_pred), 4)
                result["RMSE"] = round(np.sqrt(mean_squared_error(y_val, y_pred)), 2)
            else:
                result["R²"] = round(r2_score(y_val, y_pred), 4)
                result["RMSE"] = round(np.sqrt(mean_squared_error(y_val, y_pred)), 2)

            results.append(result)
        except Exception as e:
            results.append({"Model": name, "Error": str(e)})

    return results
