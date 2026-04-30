import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
crop_dataset = pd.read_csv("./Datasets/synthetic_55k_Dataset.csv")
market_Price_dataset = pd.read_csv("./Datasets/marketPrice.csv")
print("Crop Dataset Shape:", crop_dataset.shape)
print("Market Dataset Shape:", market_Price_dataset.shape)
crop_dataset.dropna(inplace=True)
market_Price_dataset.dropna(inplace=True)
print("Checking for duplicates in Crop Dataset:", crop_dataset.duplicated().sum())
crop_dataset = crop_dataset=crop_dataset.drop_duplicates()
market_Price_dataset = market_Price_dataset.drop(["Grade","Modal_x0020_Price"], axis=1 )
market_Price_dataset = market_Price_dataset.rename(columns={'Min_x0020_Price': 'minprice',
                                                             'Max_x0020_Price': 'maxprice',''
                                                             'Variety': 'cropname',
                                                             'Arrival_Date': 'ArrivalDate'})
crop_dataset["label"] = crop_dataset["label"].str.lower().str.strip()
market_Price_dataset["cropname"] = market_Price_dataset["cropname"].str.lower().str.strip()

market_Price_dataset['avg_price'] = (
    market_Price_dataset['minprice'] + market_Price_dataset['maxprice']
) / 2
print("\nMarket Summary:")
print(market_Price_dataset.head())
scaler = MinMaxScaler()

market_Price_dataset['demand_index'] = scaler.fit_transform(
    market_Price_dataset[['avg_price']]
)
Final_Dataset = crop_dataset.merge(
    market_Price_dataset[['cropname', 'demand_index']],
    left_on='label',
    right_on='cropname',
    how='inner'
)
Final_Dataset['demand_index'].fillna(
    Final_Dataset['demand_index'].mean(),
    inplace=True
)
features = [
    'N',
    'P',
    'K',
    'temperature',
    'humidity',
    'ph',
    'rainfall',
    'demand_index'
]

X = Final_Dataset[features]
y = Final_Dataset['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

rfmodel=RandomForestClassifier(n_estimators=100, random_state=42)
rfmodel.fit(X_train,y_train)
y_pred=rfmodel.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted') 
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("\n================ RANDOM FOREST PERFORMANCE ================")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

joblib.dump(rfmodel, 'RandomForestModel.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
