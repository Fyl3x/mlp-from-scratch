import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/complaints.csv")

data.loc[data["Product"].isin(["Credit reporting or other personal consumer reports","Credit reporting, credit repair services, or other personal consumer reports","Credit reporting"]),"Product"] = "Credit reporting"
data.loc[data["Product"].isin(["Credit card","Credit card or prepaid card"]),"Product"] = "Credit card"
data.loc[data["Product"].isin(["Checking or savings account","Bank account or service"]),"Product"] = "Bank account or service"
data.loc[data["Product"].isin(["Payday loan","Payday loan, title loan, or personal loan","Payday loan, title loan, personal loan, or advance loan"]),"Product"] = "Payday/title/personal loan"
data.loc[data["Product"].isin(["Money transfers","Money transfer, virtual currency, or money service","Virtual currency"]),"Product"] = "Money transfer / virtual currency"

data = data[data["Consumer complaint narrative"].notna()].copy()

data3 = data[data["Product"] != "Other financial service"]
data4 = data3.groupby("Product").apply(lambda x: x.sample(n=min(len(x),12000),random_state=42)).reset_index(drop=False)
data5 = data4.drop(columns=["level_1"])

training, test = train_test_split(data5,test_size=0.2,train_size=0.8,stratify=data5["Product"],random_state=42)
train, val = train_test_split(training,test_size=0.2,train_size=0.8,stratify=training["Product"],random_state=42)


train.to_csv("data/processed/train.csv",index=False)
test.to_csv("data/processed/test.csv",index=False)
val.to_csv("data/processed/val.csv",index=False)