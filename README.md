# Machine_LearningTrack_customer-churn-analysis

🔹 Business Understanding

Customer churn is a critical issue in the telecom industry. This project aims to analyze customer data and identify patterns that lead to churn in order to support retention strategies.

🔹 Dataset Inspection

The dataset contains customer demographic details, services subscribed, billing information, and churn status.

Numerical features: tenure, MonthlyCharges, TotalCharges
Categorical features: gender, Contract, InternetService, PaymentMethod

Missing values in TotalCharges were handled by converting to numeric and filling with 0.

🔹 Exploratory Data Analysis
1. Churn Distribution

Most customers did not churn, but a significant portion did, making this a relevant classification problem.

2. Contract Type vs Churn

Customers with month-to-month contracts showed higher churn compared to long-term contracts.

3. Tenure vs Churn

Customers with low tenure are more likely to churn, indicating that new customers are at higher risk of leaving.

4. Monthly Charges vs Churn

Churned customers are mostly concentrated in the mid-range monthly charges (approximately 10–70), suggesting pricing may influence early churn.

5. Correlation Heatmap

A strong positive correlation was observed between tenure and total charges, indicating that long-term customers contribute more revenue. Monthly charges also showed moderate correlation with total charges.

🔹 Key Insights
New customers are more likely to churn
Month-to-month contracts have higher churn rates
Pricing may influence churn behavior
Long-term customers generate more revenue
🔹 Machine Learning Proposal

The problem is a classification task. The following algorithms are proposed:

Logistic Regression
Decision Tree
Random Forest

Evaluation metrics:

Accuracy
Precision
Recall
F1-score
