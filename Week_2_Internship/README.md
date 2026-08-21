# Customer Churn Prediction & Retention Strategy
- Data cleaning
The central objective is to transform raw customer data into a clean and structured dataset that can be used for subsequent **customer churn prediction**.


---


# 🎯 Business Problem


Customer churn occurs when a customer stops using a company's products or services.


For a telecommunications company, customer churn can negatively affect:


- Recurring revenue
- Customer lifetime value
- Customer retention
- Business growth
- Marketing efficiency


The business problem addressed by this project is:


> **How can customer data be prepared and transformed so that machine learning can be used to identify customers who are at risk of churn?**


The long-term goal is to support a proactive, data-driven customer retention strategy.


---


# 💡 Project Objective


The objective of this preprocessing project is to transform the raw customer churn dataset into a reliable machine-learning-ready dataset.


Specifically, this project aims to:


1. Understand the structure and quality of the dataset.
2. Identify missing values and data-quality issues.
3. Correct inappropriate data types.
4. Identify duplicate records.
5. Standardize inconsistent categorical values.
6. Remove irrelevant features.
7. Engineer meaningful new features.
8. Encode categorical variables.
9. Scale numerical variables.
10. Detect potential outliers.
11. Analyze relationships between features and churn.
12. Produce a final dataset ready for machine learning.


---


# 📊 Dataset


The project uses the **Telco Customer Churn dataset**.


The original dataset contains:


| Property | Value |
|---|---:|
| Rows | 7,043 |
| Columns | 21 |
| Target Variable | `Churn` |


The dataset contains information about customer demographics, account information, subscribed services, contract type, payment methods, tenure, and billing.


---


# 🎯 Target Variable


The target variable is:


```text
Churn

It contains two possible outcomes:

Value	Meaning
No	Customer did not churn
Yes	Customer churned

For machine learning, the target was converted into a binary numerical representation:

No  → 0
Yes → 1

This allows the problem to be treated as a binary classification problem.

🗂️ Main Dataset Features

The dataset contains features from several customer-related domains.

Demographic Features
gender
SeniorCitizen
Partner
Dependents
Account & Billing Features
tenure
MonthlyCharges
TotalCharges
Contract
PaperlessBilling
PaymentMethod
Service Features
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Identifier
customerID

The customerID column was removed before the final modeling dataset because it is an identifier rather than a meaningful predictive feature.

🔍 Data Inspection

The first stage of the project was to understand the raw dataset before making any transformations.

The following were investigated:

First 10 rows
Dataset dimensions
Data types
Missing values
Duplicate records
Unique categorical values
Descriptive statistics

This inspection revealed an important issue with the TotalCharges column.

Although TotalCharges represents a numerical billing value, it was initially stored as a text/object variable.

🧹 Data Cleaning
1. TotalCharges Data-Type Correction

The TotalCharges column initially contained blank string values and was therefore represented as an object rather than a numerical variable.

The values were converted into numerical form using numeric coercion.

The blank values resulted in 11 missing numerical observations.

These records corresponded to customers with:

tenure = 0

Since these customers had zero accumulated tenure, their cumulative charges were treated as:

TotalCharges = 0.0

This allowed the variable to be used for numerical analysis and machine learning preprocessing.

2. Duplicate Record Detection

Duplicate records were checked using pandas.

The result was:

Duplicate records detected: 0

Therefore, no customer records were removed because of duplication.

3. Standardization of Inconsistent Values

Service-related categorical variables contained values such as:

Yes
No
No internet service
No phone service

For relevant service indicators, the categories:

No internet service
No phone service

were standardized to:

No

This created more consistent binary service indicators.

The standardization was applied to service-related variables including:

OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
MultipleLines

The purpose was to make these variables easier to encode and interpret during machine learning.

4. Removing the Customer Identifier

The following feature was removed:

customerID

The identifier uniquely identifies customers but does not provide meaningful behavioral information for predicting churn.

Removing it also prevents the machine learning model from treating customer identifiers as potentially useful predictive information.

🧠 Feature Engineering

Feature engineering was performed to create additional variables that could capture useful business information from the existing dataset.

Four additional features were created.

1. Total Services

A feature called:

Total Services

was created.

It represents the number of services associated with each customer.

The feature was constructed using service-related variables such as:

PhoneService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Why was it created?

Customers using a broader range of services may have a different relationship with the company compared with customers using fewer services.

Therefore, the feature provides a simplified measure of service engagement.

2. Avg_Monthly_Ratio

A feature called:

Avg_Monthly_Ratio

was created using:

Avg_Monthly_Ratio = TotalCharges / (tenure + 1)

The +1 prevents division by zero for customers with zero tenure.

Why was it created?

The feature provides an additional representation of historical customer billing relative to tenure.

3. Automatic_Payment

A binary feature called:

Automatic_Payment

was created from the customer's payment method.

The representation was:

1 → Automatic payment
0 → Non-automatic payment
Why was it created?

Payment behavior may contain useful information about customer account behavior and potential churn patterns.

Converting this information into a binary feature makes it easier for machine learning algorithms to use.

4. Has_Family

A binary feature called:

Has_Family

was created using:

Partner
Dependents

The feature represents whether the customer has a partner or dependents.

1 → Customer has a partner or dependents
0 → Customer has neither
Why was it created?

This feature provides a simplified representation of household/family association that may contain information about customer behavior.

🔢 Feature Encoding

Machine learning algorithms generally require numerical input.

Categorical variables were therefore transformed into numerical representations.

Different encoding techniques were selected based on the nature of the variables.

Target Encoding

The target variable Churn was converted from categorical values to binary values:

No  → 0
Yes → 1
Contract Encoding

The Contract feature contains an inherent ordering based on contract duration.

It was represented as:

Month-to-month → 0
One year       → 1
Two year       → 2

This preserves the increasing duration represented by the contract categories.

One-Hot Encoding

Nominal categorical variables were converted using one-hot encoding.

Examples include:

gender
Partner
Dependents
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
PaperlessBilling
PaymentMethod

drop_first=True was used to reduce redundant dummy variables.

The resulting categorical variables were represented using numerical 0/1 indicators.

📏 Feature Scaling

Numerical variables were standardized using:

StandardScaler

The following variables were scaled:

tenure
MonthlyCharges
TotalCharges
Total Services
Avg_Monthly_Ratio

Standardization uses the following transformation:

z = (x - mean) / standard deviation

This places numerical features on comparable scales.

Why StandardScaler?

Scaling is useful for machine learning algorithms that can be affected by differences in the magnitude of numerical variables.

It is particularly useful for models such as:

Logistic Regression
Support Vector Machines
K-Nearest Neighbors
Other scale-sensitive algorithms
📦 Outlier Detection

Potential outliers were investigated using the Interquartile Range (IQR) method.

The IQR is calculated as:

IQR = Q3 - Q1

The boundaries are:

Lower Bound = Q1 - 1.5 × IQR


Upper Bound = Q3 + 1.5 × IQR

The following numerical features were examined:

tenure
MonthlyCharges
TotalCharges
Total Services
Avg_Monthly_Ratio

The implemented IQR analysis detected:

0 outliers

for the selected numerical variables.

Therefore, no observations were removed as outliers.

📈 Feature Selection & Correlation Analysis

Correlation analysis was used to investigate relationships between processed features and the target variable.

Some of the stronger positive relationships with churn included:

Feature	Correlation with Churn
InternetService_Fiber optic	0.308
PaymentMethod_Electronic check	0.302
MonthlyCharges	0.193
PaperlessBilling_Yes	0.192

Some of the stronger negative relationships included:

Feature	Correlation with Churn
Encoded_Contract	-0.397
tenure	-0.352
InternetService_No	-0.228
Automatic_Payment	-0.210
TotalCharges	-0.199
Interpretation

The correlation analysis suggests that contract structure and customer tenure have particularly strong relationships with the churn target in this dataset.

However:

Correlation does not imply causation.

The results indicate associations that can be investigated further during the machine learning modeling stage.

📊 Key Business Insights

The exploratory analysis conducted during the project identified several important patterns.

Lower Tenure

Customers who churn tend to have lower tenure.

This suggests that the early stages of the customer relationship may represent an important period for customer retention.

Higher Monthly Charges

Churned customers showed higher average monthly charges.

This suggests that pricing or perceived value may be relevant factors in customer attrition.

Month-to-Month Contracts

Customers on month-to-month contracts showed the strongest churn tendency compared with customers on longer-term contracts.

These findings provide useful directions for future churn prediction and retention strategies.

🏗️ Preprocessing Workflow

The complete workflow followed this structure:

                 RAW DATASET
                     │
                     ▼
             DATA INSPECTION
                     │
                     ▼
              DATA CLEANING
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
      Missing     Data Type   Duplicate
       Values     Correction    Check
          │          │          │
          └──────────┼──────────┘
                     ▼
       STANDARDIZATION OF VALUES
                     │
                     ▼
           FEATURE ENGINEERING
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
      Total       Average     Payment/
     Services     Monthly     Family
                   Ratio
                     │
                     ▼
             FEATURE ENCODING
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Binary      Ordinal    One-Hot
       Mapping     Encoding   Encoding
                     │
                     ▼
             FEATURE SCALING
                     │
                     ▼
            OUTLIER DETECTION
                     │
                     ▼
           FEATURE SELECTION
                     │
                     ▼
          ML-READY DATASET
📁 Repository Structure
analystlab-africa-customer-churn/
│
├── README.md
│
├── notebooks/
│   └── Week_2_Customer_Churn_Preprocessing.ipynb
│
├── data/
│       └── Telco_Customer_Churn_ML_Ready.csv
│
├── reports/
│   ├── Business_Understanding_Report.pdf
│   └── Data_Preprocessing_Report.pdf
│
└── images/
    |---telco_churn_6_visualizations.png

Dataset note: If the original dataset is not permitted to be publicly redistributed, it should not be uploaded to the GitHub repository. Instead, provide instructions for obtaining the dataset.

🛠️ Technologies Used

The project was developed using:

Python
Google Colab
Jupyter Notebook
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
📦 Final Dataset

The preprocessing workflow transformed the dataset from:

7,043 rows × 21 columns

to:

7,043 rows × 27 columns

The processed dataset was saved as:

Telco_Customer_Churn_ML_Ready.csv

The final dataset contains:

Cleaned numerical variables
Encoded categorical variables
Engineered features
Scaled numerical variables
Binary churn target

It is prepared for the next stage of the project:

Machine Learning Model Development & Evaluation

📚 What I Learned

One of my biggest lessons from Week 2 was:

Machine learning does not start with the model. It starts with understanding and preparing the data.

Before this project, it was easy to think about machine learning as:

Dataset → Model → Prediction

Working through the preprocessing stage showed me that the real workflow is much more involved:

Understand
    ↓
Inspect
    ↓
Clean
    ↓
Transform
    ↓
Engineer
    ↓
Encode
    ↓
Scale
    ↓
Analyze
    ↓
Train
    ↓
Evaluate

I learned that preprocessing is not simply about "fixing" a dataset.

It requires understanding the meaning behind the data.

For example, when I discovered blank values in TotalCharges, the correct question was not simply:

"How do I delete these missing values?"

The better question was:

"Why are these values missing?"

After investigating the records, I found that they corresponded to customers with zero tenure. Understanding the business meaning of the data allowed me to make a more informed preprocessing decision.

This experience taught me that:

Good machine learning begins with good data understanding.

🔐 Data Leakage Consideration

An important lesson from this project is that preprocessing must be performed carefully to avoid data leakage.

During future model development, transformations such as scaling should be fitted only on the training data and then applied to validation/test data.

The modeling stage should therefore use an appropriate Scikit-learn pipeline where possible.

This ensures that information from the test set does not influence the training process.

🚀 Next Steps

The next stage of this project will focus on machine learning model development.

Planned steps include:

Split the dataset into training and testing sets.
Establish a baseline model.
Train multiple classification algorithms.
Evaluate model performance.
Compare models.
Investigate feature importance.
Address class imbalance if required.
Tune model hyperparameters.
Select the best-performing model.
Evaluate the final model against business requirements.

Potential algorithms include:

Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
🎓 Internship Learning Journey

This project is part of my ongoing journey through the AnalystLab Africa Machine Learning Internship Programme.

Week 1

Machine Learning Problem Framing & Data Understanding

Focus:

Business problem
Customer churn
Target definition
Data exploration
Initial business insights
Week 2

Data Preprocessing & Feature Engineering

Focus:

Data cleaning
Missing values
Data types
Feature engineering
Encoding
Scaling
Outlier detection
Feature selection
ML-ready dataset
Coming Next

Machine Learning Model Development

👨‍💻 Author
Adu Collins

Machine Learning Intern
AnalystLab Africa Machine Learning Internship Programme

Areas of Interest
Artificial Intelligence
Machine Learning
Data Science
Software Engineering
AI Engineering
Data-driven solutions
🙏 Acknowledgements

Special thanks to AnalystLab Africa for providing a practical, project-based environment for developing machine learning and data science skills.

This project provided an opportunity to move beyond theoretical machine learning concepts and work through an end-to-end data preparation workflow.
