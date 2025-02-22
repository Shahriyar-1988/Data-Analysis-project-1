import pandas as pd
import yaml
import os
import sys
from sklearn.model_selection import train_test_split

# load yaml parameters
params = yaml.safe_load(open('params.yaml'))["preprocess"]

def preprocessor(input_path,output_path):
    df = pd.read_csv(input_path)
    # Step1: Initial cleaning of data
    df.TotalCharges=pd.to_numeric(df.TotalCharges,errors='coerce')
    df.dropna(axis=0,inplace=True)
    df["Churn"]=df["Churn"].map({'Yes':1,'No':0}) # Converting Yes/No into binary values
    redundant_col=["MultipleLines",
               "OnlineSecurity",
               "OnlineBackup",
               "DeviceProtection",
               "TechSupport"] # Columns containing redundant values
    
    for col in redundant_col:
        df[col]=df[col].where(df[col].isin(["Yes","No"]),"No")
    # Step2: Drop any unnecessary
    df1 = df.drop(["gender",'StreamingMovies',"StreamingTV","TotalCharges"],axis=1)
    try:
        # Check for null values
        if df1.isnull().sum().sum()> 0:
            raise ValueError("Error: Data contains missing values. Please clean the dataset before proceeding.")

    except ValueError as e:
        print(e)  # Print the error message
        sys.exit(1)  # Terminate execution
    # Step3: Reduce the dimensions where possible
    df1['PaymentMethod']=df1['PaymentMethod'].where(df1['PaymentMethod']=='Electronic check','Others')
    # Step 4: Split data
    train_data,test_data = train_test_split(df1,test_size=0.2,stratify=df1[["Churn"]])
    # Step 5: Create data path
    train_path=output_path["train"]
    test_path = output_path['test']
    
    os.makedirs(os.path.dirname(train_path),exist_ok=True)
    os.makedirs(os.path.dirname(test_path),exist_ok=True)
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    print(f"Training data saved to {train_path} successfully!")
    print(f"Test data saved to {test_path} successfully!")

if __name__=='__main__':
    input_path,output_path = params["input"],params["output"]
    preprocessor(input_path,output_path)




    

