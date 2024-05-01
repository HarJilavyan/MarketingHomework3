# Customer Churn Prediction Using Survival Analysis

## Project Overview

This project applies survival analysis techniques to predict customer churn for a telecommunications company. Using a dataset that includes various customer attributes such as tenure, age, income, marital status, and service usage patterns, we build and compare several survival models to estimate the time until churn.

## Objective

The main goal of this project is to understand and predict when a customer will churn, enabling proactive customer retention strategies. By identifying key factors that influence churn and assessing the risk and timing of churn, the company can implement targeted interventions to improve customer retention and overall profitability.

## Models Used

- **Weibull AFT Model**: Provides a baseline survival model with interpretable parameters, ideal for capturing time-to-event data with increasing or decreasing failure rates over time.
- **Log-Normal AFT Model**: Used for data where the event log-time is normally distributed, allowing for a more flexible fit that can accommodate variable churn rates.
- **Log-Logistic AFT Model**: Suitable for cases where the churn rate increases initially and decreases over time, offering a versatile fit across different customer behaviors.

## Data

The dataset includes the following key features:
- **ID**: Unique identifier for each customer.
- **Tenure**: Duration of the customer's relationship with the company.
- **Age**: Customer's age.
- **Marital Status**: Indicates whether the customer is married.
- **Income**: Annual income of the customer.
- **Services Used**: Types of services the customer has subscribed to (e.g., voice, internet).

Data preprocessing steps include one-hot encoding of categorical variables and conversion of the churn variable to a binary format for survival analysis.

## Analysis

- **Model Fitting**: Each model is fitted to the dataset to estimate survival functions and the impact of covariates on churn risk.
- **Model Evaluation**: Comparisons are made based on statistical criteria and model performance metrics such as AIC, BIC, and concordance index.
- **Visualization**: Survival curves for each model are plotted to visualize and compare the estimated survival probabilities over time.

## Customer Lifetime Value (CLV)

- **Calculation**: CLV is calculated based on the expected tenure derived from the survival models and the average revenue per customer.
- **Usage**: CLV is used to prioritize retention efforts, focusing on high-value customer segments that are at risk of churning.

## Retention Strategies Recommended

- **Personalized Marketing**: Targeting customers based on their risk of churn and preferences.
- **Loyalty Programs**: Implementing loyalty incentives for high-risk segments to enhance customer satisfaction and retention.
- **Customer Service Enhancements**: Improving customer service interactions based on feedback and churn predictors.

## How to Run This Project

Instructions for setting up and running the project locally, including installation of required software and libraries, loading the dataset, and executing the analysis scripts.

## Contributing

Guidelines for contributing to this project, including code style, submission process, and other requirements.

---

For more information on the analysis methods, model details, or to get involved, please contact [Your Contact Information].
