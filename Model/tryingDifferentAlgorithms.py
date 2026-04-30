import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
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
from sklearn.svm import SVC
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
market_Price_dataset = market_Price_dataset.drop(["Grade","Modal_x0020_Price"], axis=1 )
market_Price_dataset = market_Price_dataset.rename(columns={'Min_x0020_Price': 'minprice', 'Max_x0020_Price': 'maxprice','Variety': 'cropname','Arrival_Date': 'ArrivalDate'})
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
print("Checking for null values in Final_Dataset:")
print(Final_Dataset.isnull().sum())
print(Final_Dataset.shape)
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
sns.heatmap(
    Final_Dataset[
        ['N','P','K','temperature',
         'humidity','ph','rainfall',
         'demand_index']
    ].corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
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
lgmodel=LogisticRegression()
lgmodel.fit(X_train,y_train)
y_pred=lgmodel.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted') 
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("\n================ LOGISTIC REGRESSION PERFORMANCE ================")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

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
dtmodel=DecisionTreeClassifier(random_state=42)
dtmodel.fit(X_train,y_train)
y_pred=dtmodel.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted') 
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("\n================ DECISION TREE PERFORMANCE ================")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

xgbmodel = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    eval_metric='mlogloss',
    random_state=42
)

xgbmodel.fit(X_train, y_train)
y_pred = xgbmodel.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n================ XGB MODEL PERFORMANCE ================")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

ada_model = AdaBoostClassifier()
ada_model.fit(X_train, y_train)
y_pred = ada_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n================ ADA BOOST MODEL PERFORMANCE ================")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# joblib.dump(model, 'XGBClassifierModel.pkl')
# joblib.dump(label_encoder, 'label_encoder.pkl')

# ================= MODEL COMPARISON TABLE =================

results = []

# Logistic Regression
results.append({
    "Algorithm": "Logistic Regression",
    "Accuracy": accuracy_score(y_test, lgmodel.predict(X_test)),
    "F1-Score": f1_score(
        y_test,
        lgmodel.predict(X_test),
        average='weighted'
    )
})

# Random Forest
results.append({
    "Algorithm": "Random Forest",
    "Accuracy": accuracy_score(y_test, rfmodel.predict(X_test)),
    "F1-Score": f1_score(
        y_test,
        rfmodel.predict(X_test),
        average='weighted'
    )
})

# Decision Tree
results.append({
    "Algorithm": "Decision Tree",
    "Accuracy": accuracy_score(y_test, dtmodel.predict(X_test)),
    "F1-Score": f1_score(
        y_test,
        dtmodel.predict(X_test),
        average='weighted'
    )
})

# XGBoost
results.append({
    "Algorithm": "XGBoost",
    "Accuracy": accuracy_score(y_test, xgbmodel.predict(X_test)),
    "F1-Score": f1_score(
        y_test,
        xgbmodel.predict(X_test),
        average='weighted'
    )
})

# AdaBoost
results.append({
    "Algorithm": "AdaBoost",
    "Accuracy": accuracy_score(y_test, ada_model.predict(X_test)),
    "F1-Score": f1_score(
        y_test,
        ada_model.predict(X_test),
        average='weighted'
    )
})

# Convert into DataFrame
results_df = pd.DataFrame(results)

# Convert values into percentage format
results_df["Accuracy"] = results_df["Accuracy"].apply(lambda x: round(x * 100, 2))
results_df["F1-Score"] = results_df["F1-Score"].apply(lambda x: round(x * 100, 2))

# Print table
print("\n================ MODEL COMPARISON TABLE ================\n")
print(results_df.to_string(index=False))

print("\n" + "=" * 58)
print("| {:^20} | {:^12} | {:^12} |".format(
    "Algorithm", "Accuracy", "F1-Score"
))
print("=" * 58)

for index, row in results_df.iterrows():
    print("| {:<20} | {:^12} | {:^12} |".format(
        row["Algorithm"],
        str(row["Accuracy"]) + "%",
        str(row["F1-Score"]) + "%"
    ))
    print("-" * 58)
    plt.figure(figsize=(10,8))

sns.heatmap(
    final_dataset[
        ['N','P','K','temperature',
         'humidity','ph','rainfall',
         'demand_index']
    ].corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()