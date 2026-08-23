# 📊 Telecommunications Customer Churn Prediction & Retention Strategy

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?logo=scikit-learn)
![Framework](https://img.shields.io/badge/Framework-CRISP--DM-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end Machine Learning project developed as part of the **AnalystLab Africa Machine Learning Internship Programme (Week 1 – Week 3)**. This repository contains the complete workflow for predicting customer churn for **ABC Communications Ltd.**, spanning business problem framing, feature engineering, statistical analysis, model development, multi-algorithm evaluation, and business intervention strategies.

---

## 📌 Business Overview & Objective

In the telecommunications sector, acquiring a new customer is **5 to 7 times more expensive** than retaining an existing subscriber. ABC Communications Ltd. previously operated on a *reactive* retention model—engaging customers only after contract termination requests were submitted.

### Project Goals
1. Transition from a reactive to a **proactive retention model**.
2. Identify high-risk churn subscribers weeks before contract expiration.
3. Optimize retention campaign ROI by targeting customers based on risk probability and key behavioral drivers.

---

## 📐 Machine Learning Problem Formulation

* **Task Type:** Supervised Binary Classification
* **Target Variable:** `Churn` (`1` = Churned, `0` = Retained)
* **Dataset Size:** 7,043 Customer Records | 21 Features
* **Primary Optimization Metric:** **Recall (Sensitivity)** — Minimizing False Negatives (unidentified churners) is prioritized over raw accuracy because losing a customer results in direct cumulative revenue loss.

---

## 🛠️ Project Repository Structure

```text
.
├── data/
│   └── telco_churn_processed.csv       # Machine learning-ready clean dataset
├── docs/
│   ├── Business_Understanding_Report.pdf
│   ├── Data_Preprocessing_Report.pdf
│   ├── Model_Evaluation_Report.pdf
│   ├── Feature_Importance_Report.pdf
│   └── AnalystLab_Africa_Week3_Documentation_Reports.docx
├── notebooks/
│   └── Customer_Churn_ML_Pipeline.ipynb # Main end-to-end Jupyter Notebook
├── visualizations/
│   ├── viz_1_to_4_confusion_matrices.png
│   ├── viz_5_roc_curves.png
│   ├── viz_6_precision_recall_curves.png
│   ├── viz_7_model_comparison_bar.png
│   ├── viz_8_feature_importance.png
│   ├── viz_9_probability_distribution.png
│   └── viz_10_learning_curve.png
├── .gitignore
├── README.md
└── requirements.txt
⚙️ Data Preprocessing & Feature Engineering SummaryMissing Value Imputation: Coerced whitespace strings in TotalCharges into NaN values (customers with tenure = 0) and imputed them with 0.0.Identifier Dropping: Stripped customerID to eliminate non-predictive high-cardinality noise.Engineered Features:Avg_Monthly_Ratio: Calculated $\text{TotalCharges} / (\text{tenure} + 1)$ with zero-bias handling.Total_Services: Aggregated active service subscriptions to quantify customer engagement/stickiness.Automatic_Payment & Has_Family: Created binary indicators for payment convenience and family presence.Categorical Encoding & Scaling: Binary mapping for binary columns, One-Hot Encoding (drop_first=True) for multi-class variables, and StandardScaler for continuous numeric features.🔬 Model Benchmarking & Evaluation ResultsFour classification models representing distinct machine learning paradigms were trained using a Stratified 80/20 Train-Test Split ($N_{\text{train}} = 5,634$, $N_{\text{test}} = 1,409$) with class-weight balancing applied:Algorithm ModelAccuracyPrecisionRecall (Primary)F1 ScoreROC-AUC🏆 Logistic Regression0.74380.51130.78610.61960.8475🌲 Random Forest0.77220.56240.63900.59820.8252⚡ Support Vector Machine (SVM)0.78920.59800.62830.61280.8289🌴 Decision Tree0.73310.49730.49200.49460.6566Metric Key InsightsThe Accuracy Trap: While SVM achieved the highest Accuracy ($78.92\%$), it missed 37.2% of actual churners ($\text{Recall} = 62.83\%$).Recommended Model: Logistic Regression delivered the highest Recall ($78.61\%$) and ROC-AUC ($0.8475$), capturing 294 out of 374 actual churners in the test set.💡 Key Drivers of Customer ChurnModel coefficient analysis highlights critical behavioral patterns governing subscriber churn:🚨 Top Churn Drivers (Positive Coefficients):Contract_Month-to-month: High flight risk due to zero switching friction.InternetService_Fiber optic: High pricing sensitivity combined with service outages.PaymentMethod_Electronic check: High payment manual friction.🛡️ Top Retention Drivers (Negative Coefficients):tenure: Longer tenure strongly builds brand stickiness.Contract_Two year: Long-term contractual locking.TechSupport_Yes & OnlineSecurity_Yes: Value-added support features act as retention anchors.🎯 Business Recommendations & Next StepsDeployment: Deploy the trained Logistic Regression model pipeline to generate daily churn risk probabilities for marketing decision support.Targeted Campaigns: Prioritize Month-to-Month subscribers with $<12$ months tenure who lack technical support add-ons.Contract Conversion Offers: Provide promotional discounts (e.g., 3 months free tech support) to incentivize switching from Month-to-Month to 1-Year or 2-Year contracts.Optimal Thresholding: Evaluate custom classification thresholds below $0.50$ to push Recall toward $\ge 80\%$.🚀 Quickstart GuideClone the Repository:Bashgit clone [https://github.com/your-username/telco-customer-churn-prediction.git](https://github.com/your-username/telco-customer-churn-prediction.git)
cd telco-customer-churn-prediction
Create a Virtual Environment & Install Dependencies:Bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Run the Jupyter Notebook:Bashjupyter notebook notebooks/Customer_Churn_ML_Pipeline.ipynb
👤 Author & AcknowledgmentsDeveloper: Junior Machine Learning EngineerOrganization: AnalystLab Africa — Machine Learning Internship ProgrammeHashtag: #AnalystLabAfrica
