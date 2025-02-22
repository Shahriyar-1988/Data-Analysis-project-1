# Project Summary: Investigating Imbalanced Data in ML Models

In this project, I aim to investigate the impact of imbalanced data on machine learning model performance and devise effective strategies to mitigate this issue. Specifically, I will explore the efficacy of simple down-sampling, SMOTE (Synthetic Minority Over-sampling Technique), and ADASYN (Adaptive Synthetic Sampling) methods. The project will focus on a dataset concerning customer churn. The key steps involved are outlined below:

1. **Dataset Import:** Begin by importing the imbalanced dataset of customer churn.

2. **Data Discovery and EDA:** Conduct thorough data discovery and exploratory data analysis (EDA) to gain insights into the dataset's features, their trends, and their influence on the target value.

3. **Model Training - Logistic Regression:**
   - Train a logistic regression model on the dataset under three different scenarios:
     1. Training on the imbalanced data to establish baseline performance.
     2. Training on data after simple down-sampling, SMOTE over-sampling, and ADASYN synthesis to compare results.
4. **XGBoost:**
 Finally train the best-performing scenario from the previous step on a powerful XGBoost classifier to push the performance limits on this dataset.

By following this structured approach, I aim to uncover valuable insights into handling imbalanced data and optimizing machine learning models for improved performance.