Author
Thanh Dat Nguyen
Applied Artificial Intelligence & Machine Learning
# 🔮 Telco Customer Churn Prediction: End-to-End ML Pipeline

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://[LINK-STREAMLIT-CỦA-BẠN])
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.0-orange.svg)](https://xgboost.readthedocs.io/)

## 📌 Business Objective
Customer retention is a critical metric for subscription-based businesses. Acquiring a new customer is significantly more expensive than retaining an existing one. This project utilizes machine learning to:
* Proactively identify customers at high risk of churning.
* Provide actionable, personalized retention strategies based on Model Explainability (SHAP).
* Enable bulk evaluation for marketing teams to process large datasets efficiently.

## 🚀 Live Demo
**Interact with the deployed web application here:** [Insert Your Streamlit Cloud Link]

## ✨ Key Features
* **Individual Risk Assessment:** Real-time prediction interface with Plotly gauge charts visualizing churn probability.
* **Explainable AI (XAI):** Integrated SHAP (SHapley Additive exPlanations) waterfall plots to decode the XGBoost "black box", revealing the exact impact of each customer feature (e.g., Tenure, Contract Type) on the final prediction.
* **Enterprise Batch Prediction:** Seamless CSV upload processing, allowing bulk scoring and automated risk categorization with one-click report downloading.
* **Dynamic Recommendations:** Rule-based business logic providing tailored retention strategies based on real-time model outputs.

## 🧠 Technical Architecture & Pipeline

| Phase | Techniques Used | Description |
| :--- | :--- | :--- |
| **Data Engineering** | Missing Value Imputation, One-Hot Encoding | Cleaned raw Telco data and mapped categorical variables into numerical matrices suitable for tree-based algorithms. |
| **Data Balancing** | SMOTE | Handled the natural 73:27 class imbalance to prevent the model from becoming biased toward the majority class (No Churn). |
| **Model Training** | XGBoost, Random Forest | Trained and evaluated models focusing on **Recall**, prioritizing the minimization of False Negatives (missing a churning customer). |
| **XAI Integration** | SHAP | Extracted global feature importance and local instance-level explanations for transparent AI decision-making. |
| **Deployment** | Streamlit, Joblib | Serialized the trained model and column structures, serving them through a responsive web interface. |

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Yesod2810/telco-churn-prediction.git](https://github.com/Yesod2810/telco-churn-prediction.git)
   cd telco-churn-prediction
2. **Create a virtual environment and install dependencies:**
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
3. **Run the Streamlit application**
  cd src
  streamlit run app.py
