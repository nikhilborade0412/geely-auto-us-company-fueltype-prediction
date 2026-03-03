
# 🚗 Geely Auto – US Market Fuel Type Prediction

### Logistic Regression | SMOTE | Streamlit | Business Strategy Project

---

## 📌 Project Overview

Geely Auto plans to enter the US automobile market and needs data-driven insights to understand customer fuel type preferences (Gas vs Diesel).

This project builds a **Logistic Regression model** to predict whether a vehicle uses:

* **Gas (0)**
* **Diesel (1)**

The analysis helps identify key factors influencing fuel type selection and provides strategic recommendations for market entry.

---

## 🎯 Business Problem

Fuel type selection directly impacts:

* Engine design
* Pricing strategy
* Performance positioning
* Market segmentation

Geely Auto requires a predictive model to understand which vehicle specifications influence fuel choice in the US market.

---

## 🧠 Business Objective

* Identify significant factors affecting fuel type selection
* Handle class imbalance using SMOTE
* Build a reliable classification model
* Provide strategic recommendations for US market entry

---

## 📊 Dataset Overview

* **Dataset Name:** CarPrice_Assignment.csv
* **Total Records:** 205 vehicles
* **Total Features:** 26 columns
* **Target Variable:** `fueltype`

### Key Features:

* Engine Size
* Compression Ratio
* Horsepower
* City MPG / Highway MPG
* Price
* Drivewheel
* Carbody

The dataset was clean with:

* ✅ No missing values
* ✅ No duplicates

---

## ⚙️ Project Workflow

### 1️⃣ Data Cleaning

* Removed irrelevant columns (`car_ID`, `CarName`)
* Encoded categorical variables
* Converted target variable (Gas=0, Diesel=1)

### 2️⃣ Exploratory Data Analysis

* Univariate & Bivariate analysis
* Identified class imbalance

### 3️⃣ Data Preprocessing

* Train-Test Split (Stratified)
* Applied **SMOTE** on training data
* Standardized numerical features

### 4️⃣ Model Building

* Logistic Regression
* Probability-based prediction

### 5️⃣ Model Evaluation

* Accuracy Score
* Confusion Matrix
* Classification Report
* ROC Curve & AUC Score

---

## 📈 Key Insights

* Diesel vehicles are associated with:

  * Higher compression ratio
  * Larger engine size
  * Higher horsepower
  * Slightly premium pricing

* Gas vehicles dominate the mass-market segment.

---

## 💼 Strategic Recommendations for Geely Auto

### ✅ Focus on Gas Vehicles

* Capture high-volume US demand
* Target urban and family customers

### ✅ Selective Diesel Positioning

* Introduce diesel variants in mid-to-premium SUVs
* Emphasize durability and highway efficiency

### ✅ Pricing Strategy

* Competitive pricing for gas vehicles
* Premium positioning for diesel models

---

## 📂 Project Structure

```
GEELY_AUTO_US_MARKET_FUELTYPE/
│
├── dataset/
│   └── CarPrice_Assignment.csv
│
├── notebook/
│   └── CarPrice_Assignment.ipynb
│
├── model/
│   └── model.py
│
├── pkl/
│   ├── fueltype_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── observation & recommendation/
│
├── Report/
│   └── Geely_Auto_US_Market_Consulting_Report.pdf
│
├── video/
│   └── Recording.mp4
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🖥️ Streamlit Application

The project includes a deployed Streamlit application that:

* Takes vehicle specifications as input
* Applies preprocessing
* Predicts fuel type
* Displays prediction probability

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Model & Artifacts

* `fueltype_model.pkl` → Trained Logistic Regression model
* `scaler.pkl` → StandardScaler object
* `feature_columns.pkl` → Required input feature order

---

## 📊 Evaluation Performance

Model evaluated using:

* Confusion Matrix
* ROC Curve
* AUC Score

SMOTE improved minority class (Diesel) prediction performance.

---

## 🚀 Business Impact

This project enables:

* Data-driven vehicle design decisions
* Market-aligned product portfolio planning
* Reduced entry risk in US automobile market
* Strategic fuel-type positioning

---

## 👨‍💻 Author

**Nikhil Borade**
AI/ML Engineer | Data Scientist

🔗 [GitHub](https://github.com/nikhilborade0412)  |  
🔗 [LinkedIn](https://www.linkedin.com/in/nikhilborade0412/)  |  
🔗 [Portfolio](https://nikhilborade0412.github.io/)

---
