# Credit Card Fraud Detection Model



## 1. Introduction
This project develops a machine learning model to detect fraudulent credit card transactions using simulated transaction data from the western United States. The solution addresses the critical need for real-time fraud detection in financial systems.

## 2. Background
Credit card fraud costs the global economy billions annually. Traditional rule-based systems often fail to detect sophisticated fraud patterns. This project leverages machine learning to identify fraudulent transactions with higher accuracy than conventional methods.

## 3. Problem Statement
- **Challenge**: Detect fraudulent transactions in highly imbalanced data (fraud represents only ~0.58% of transactions)
- **Objective**: Build a model that maximizes fraud detection In a Transactions.

## 4. Dataset Analysis
### Dataset Overview
- Simulated transactions from 1/1/2020 to 12/31/2020
- 1000 customers, 800 merchants
- 324,169 total transactions (1,877 fraudulent)

### Key Features
- Transaction details (amount, category, timestamp)
- Geographic data (merchant/customer locations)
- Demographic information (age, gender)

### EDA Findings
- **Temporal Patterns**:
  - Highest fraud rates: 12AM-6AM (1.03%) and 6PM-12AM (1.04%)
- **Amount Analysis**:
  - Most fraud occurs in ">$150" and "$10-40" ranges
- **Category Analysis**:
  - Shopping and grocery categories most vulnerable
- **Demographics**:
  - Customers >60 years most susceptible

## 5. Tools and Technology
- **Python Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **Experiment Tracking**: Comet ML
- **Development Tools**: VS Code, Jupyter Notebooks
- **Version Control**: GitHub

## 6. Process Flow
1. **Data Preprocessing**:
   - Feature engineering (geographic distances, time buckets)
   - Bucketization of transaction amounts
   - One-hot encoding for categorical variables

2. **Feature Selection**:
   - Chi-square tests identified significant features
   - VIF analysis ensured no multicollinearity

3. **Model Development**:
   - Baseline: Logistic Regression
   - Advanced: Random Forest with HalvingGridSearchCV

4. **Evaluation**:
   - Focused on recall and F1-score for fraud class
   - Confusion matrix analysis

![Fraud Detection Vaiable Importance](image.png)

## 7. Results
### Model Performance Comparison

| Model                | Accuracy | Fraud Recall | Fraud F1 | ROC-AUC |
|----------------------|----------|--------------|----------|---------|
| Logistic Regression  | 86.72%   | 90.68%       | 7.33%    | 0.887   |
| Random Forest        | 97.79%   | 70.64%       | 26.99%   | 0.843   |

**Key Insights**:
- Random Forest improved fraud F1-score by 3.7x
- Maintained high accuracy (97.9%)
- Balanced precision/recall trade-off

## 8. Practical Applications
- **Real-time Fraud Detection**: Integrate with payment gateways
- **Risk Scoring**: Flag high-risk transactions for manual review
- **Customer Protection**: Proactively block suspicious transactions

## 9. Conclusion
The Random Forest model demonstrated superior performance in detecting fraudulent transactions while maintaining high accuracy on legitimate transactions. The use of HalvingGridSearchCV enabled efficient hyperparameter tuning without compromising model quality.

## 10. Key Learnings
- Importance of feature engineering (geographic distances, time features)
- Effectiveness of bucketing continuous variables
- Value of experiment tracking with Comet ML
- Challenges of class imbalance in fraud detection
- Superiority of ensemble methods for this problem

## 11. Future Improvements
1. **Model Enhancement**:
   - Experiment with XGBoost and Neural Networks
   - Implement anomaly detection techniques

2. **Feature Engineering**:
   - Add behavioral patterns (spending habits)
   - Incorporate network features (transaction graphs)

3. **Deployment**:
   - Create API endpoint for real-time predictions
   - Develop dashboard for fraud analysts

4. **Continuous Learning**:
   - Implement model retraining pipeline
   - Add feedback loop from fraud investigators

## Getting Started
```bash
git clone https://github.com/dushyantver/Credit-Card-Fraud-Detection-Model.git
pip install -r requirements.txt
python training_pipeline.py