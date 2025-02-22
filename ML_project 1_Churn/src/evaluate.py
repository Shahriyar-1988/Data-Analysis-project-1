import pandas as pd
import pickle
import os
import yaml
import mlflow
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score

# Load evaluation parameters from yaml
params=yaml.safe_load(open('params.yaml'))["evaluate"]

def load_model(model_path):
    with open(model_path,'rb') as file:
        return pickle.load(file)
def load_transformer(transformer_path):
    with open(transformer_path,'rb') as file:
        return pickle.load(file)
def data_processor(X,transformer_path):
    transformer = load_transformer(transformer_path)
    return transformer.transform(X)
# model logging function
def model_logger(model,X_test,y_test, model_name ):
    y_pred=model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    # Data logging
    load_dotenv()
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

    # **Set MLflow tracking URI**
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # **Create a new experiment**
    mlflow.set_experiment(f"{model_name} Evaluation")
    with mlflow.start_run():
         mlflow.log_metric('Test accuracy:',round(accuracy,3))
         mlflow.log_metric('Test f1-score:',round(f1,3))

         
         if hasattr(model, "predict_proba"):  # Ensure model supports probability predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_score=roc_auc_score(y_test,y_pred_proba)
            mlflow.log_metric('Test roc_auc score:',round(roc_score,3))

    print(f"{model_name} model test performance was successfully logged into MLFLOW!")

def model_evaluate(data,transformer_path,logistic_model_path,xgb_model_path):
    df=pd.read_csv(data)
    X_test=df.drop(["Churn"],axis=1)
    y_test=df["Churn"]
    X_test_transformed = data_processor(X_test,transformer_path)
    lr_model=load_model(logistic_model_path)
    model_logger(lr_model, X_test_transformed, y_test, "Logistic Regression")
    xg_model = load_model(xgb_model_path)
    model_logger(xg_model, X_test_transformed, y_test, "XGBoost Model")

if __name__=="__main__":
    model_evaluate(params["data"],params["transformer_path"],params["logistic_model_path"],params["xgboost_model_path"])





