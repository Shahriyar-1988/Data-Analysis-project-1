from sklearn.linear_model import LogisticRegression
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import os
import yaml
import pickle
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,roc_curve,auc
import mlflow
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from imblearn.over_sampling import ADASYN


# **Load parameters**
params = yaml.safe_load(open("params.yaml"))["train"]


# **Logistic Regression Model**
def log_fit(X_train, y_train):
    model = LogisticRegression(solver="newton-cg", random_state=32)
    return model.fit(X_train, y_train)


# **XGBoost Model with Hyperparameter Tuning**
def boost_fit(X_train, y_train, param_grid):
    xgb = XGBClassifier()
    scoring_ = ["accuracy", "precision", "recall", "f1"]
    model = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring=scoring_,
        refit="f1",
        cv=4,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model.best_estimator_, model.best_params_  # **Returns best model & params**


# **Function to Log Model & Metrics in MLflow**
def log_model_mlflow(model, X_val, y_val, model_name,accuracy_train, extra_params=None):
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    cl_rep = classification_report(y_val, y_pred, output_dict=True)
    cm = confusion_matrix(y_val, y_pred)
    # Log mlflow credintials
    # Refer to README.md to find how to handle this
    load_dotenv()
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
    MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

    # **Set MLflow tracking URI**
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # **Log MLflow experiment**
    mlflow.set_experiment(model_name)
    with mlflow.start_run():
        signature = infer_signature(X_val, y_val)
        mlflow.log_metric("Validation accuracy", round(accuracy,2))
        mlflow.log_metric("Training accuracy",round(accuracy_train,2))

        # **Log classification report metrics**
        for label, metrics in cl_rep.items():
            if isinstance(metrics, dict):
                for metric_name in ["precision", "recall", "f1-score"]:
                    mlflow.log_metric(f"{label}_{metric_name}", round(metrics[metric_name], 2))

        # **Save and log confusion matrix**
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        _, ax = plt.subplots(figsize=(6, 4))
        disp.plot(ax=ax, cmap="Blues", colorbar=True)
        plt.title(f"{model_name} - Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")
        # ** Calculate and log AUC-ROC curve(if possible)
        if hasattr(model,"predict_proba"):
            y_pred_proba = model.predict_proba(X_val)[:, 1]  # Get probabilities for ROC curve
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            # **Plot ROC Curve**
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{model_name} - ROC Curve")
            plt.legend(loc="lower right")
            plt.savefig("roc_curve.png")
            plt.close()
            mlflow.log_artifact("roc_curve.png")


        # **Log the model**
        mlflow.sklearn.log_model(model, artifact_path=model_name)

        # **Log additional parameters (if any)**
        if extra_params:
            for param, value in extra_params.items():
                mlflow.log_param(param, value)


# **Main Training Function**
def train(data, model_path, transformer_path, param_grid):
    df = pd.read_csv(data)
    X, y = df.drop("Churn", axis=1), df["Churn"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    obj_col = X.select_dtypes(include="object").columns
    num_col = ["tenure", "MonthlyCharges"]

    ct = ColumnTransformer(
        transformers=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"), obj_col),
            ("scaler", StandardScaler(), num_col),
        ],
        remainder="passthrough",
    )

    X_train = ct.fit_transform(X_train)
    X_val = ct.transform(X_val)
    os.makedirs(os.path.dirname(transformer_path), exist_ok=True)
    with open(transformer_path, "wb") as file:
        pickle.dump(ct, file)

    # **Apply ADASYN for imbalance handling**
    X_train_resampled, y_train_resampled = ADASYN(sampling_strategy="auto", random_state=0).fit_resample(
        X_train, y_train
    )

    # **Train Logistic Regression**
    lr_model = log_fit(X_train_resampled, y_train_resampled)
    accuracy_train=accuracy_score(lr_model.predict(X_train_resampled),y_train_resampled)
    log_model_mlflow(lr_model, X_val, y_val, "Logistic Regression",accuracy_train, {"solver": lr_model.solver})

    # **Save Logistic Regression Model**
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path.replace(".pkl", "_logistic.pkl"), "wb") as file:
        pickle.dump(lr_model, file)
    print(f"Logistic Regression model saved at {model_path.replace(".pkl", "_logistic.pkl")}")

    # **Train XGBoost Model**
    xg_model, xg_params = boost_fit(X_train_resampled, y_train_resampled, param_grid)
    accuracy_train=accuracy_score(xg_model.predict(X_train_resampled),y_train_resampled)
    log_model_mlflow(xg_model, X_val, y_val, "XGBoost Model",accuracy_train, xg_params)

    # **Save XGBoost Model**
    with open(model_path.replace(".pkl", "_xgboost.pkl"), "wb") as file:
        pickle.dump(xg_model, file)
    print(f"XGBoost model saved at {model_path.replace('.pkl', '_xgboost.pkl')}")


if __name__ == "__main__":
    train(data=params["data"], model_path=params["model_path"],transformer_path=params["transformer_path"],param_grid=params["param_grid"])
