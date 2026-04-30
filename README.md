# 🌾 Smart-Agriculture-Crop-Recommendation-Using-Environmental-and-Market-Price-Data
A Machine Learning based agricultural recommendation system that predicts the most suitable crop using soil nutrients, environmental conditions, and market demand analysis.

The project combines crop cultivation data with agricultural market pricing data to improve crop prediction accuracy and support better farming decisions.

---

# ✨ Features

- 🌱 Intelligent crop prediction system
- 📊 Market demand analysis using pricing data
- 🧹 Data preprocessing and cleaning
- 🔍 Feature engineering with demand index generation
- 🤖 Random Forest based classification model
- 📈 Performance evaluation using ML metrics
- 💾 Model saving using Joblib

---

# 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn |
| Model Serialization | Joblib |

---

# 📂 Project Structure

```plaintext
Smart-Crop-Prediction/
│
├── Datasets/
│   ├── synthetic_55k_Dataset.csv
│   └── marketPrice.csv
│
├── RandomForestModel.pkl
├── label_encoder.pkl
├── main.py
├── requirements.txt
└── README.md
```

---

# 📁 Dataset Information

## Crop Dataset

The crop dataset contains agricultural and environmental attributes.

| Feature | Description |
|---|---|
| N | Nitrogen value |
| P | Phosphorus value |
| K | Potassium value |
| temperature | Temperature level |
| humidity | Humidity percentage |
| ph | Soil pH value |
| rainfall | Rainfall amount |
| label | Crop name |

---

## Market Dataset

The market dataset contains crop pricing information.

| Feature | Description |
|---|---|
| cropname | Name of the crop |
| minprice | Minimum market price |
| maxprice | Maximum market price |

---

# ⚙️ Project Workflow

## 1. Data Collection

The system uses:
- Crop cultivation dataset
- Agricultural market price dataset

Both datasets are loaded and processed for training.

---

## 2. Data Cleaning

The preprocessing stage includes:
- Removing missing values
- Removing duplicate records
- Standardizing column names
- Formatting crop names consistently

This improves data quality and model performance.

---

## 3. Market Demand Analysis

The project calculates:
- average market price
- normalized demand index

The demand index helps identify crops with better market demand.

---

## 4. Dataset Merging

Both datasets are merged using crop names to combine:
- agricultural data
- environmental data
- market demand information

into a unified dataset.

---

# 📌 Features Used for Prediction

The model uses the following parameters:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall
- Demand Index

---

# 🤖 Machine Learning Model

## Random Forest Classifier

The project uses a Random Forest Classification algorithm for crop prediction.

### Why Random Forest?

- Handles large datasets efficiently
- Reduces overfitting
- Provides high prediction accuracy
- Works well for classification tasks

---

# ✂️ Train-Test Split

The dataset is divided into:
- 80% Training Data
- 20% Testing Data

This helps evaluate the model on unseen data.

---

# 🔠 Label Encoding

Crop labels are converted into numerical values before model training.

This allows the machine learning model to process categorical crop names.

---

# 📈 Model Evaluation

The system evaluates performance using:

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall prediction correctness |
| Precision | Measures correct positive predictions |
| Recall | Measures ability to identify correct crops |
| F1 Score | Balances precision and recall |

---

# 💾 Model Saving

The trained model and label encoder are saved for future prediction usage.

Generated files:

| File | Description |
|---|---|
| RandomForestModel.pkl | Trained machine learning model |
| label_encoder.pkl | Encoded crop label mappings |

---

# ▶️ Installation Guide

## Step 1: Create Virtual Environment

Create and activate a Python virtual environment.

---

## Step 2: Install Required Libraries

Install all required Python dependencies.

---

## Step 3: Run the Project

Execute the main Python file to train and evaluate the model.

---

# 📊 Output

After execution, the project displays:
- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report

The trained model is automatically stored for future use.

---
