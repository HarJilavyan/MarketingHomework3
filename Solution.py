import pandas as pd
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
import matplotlib.pyplot as plt


#  STEP 1: LOADING AND PREPARING DATASET
# Load the dataset
data_path = 'telco.csv'
telco_data = pd.read_csv(data_path)
# Display the first few rows of the dataset and the data types of the columns
print(telco_data.head(), telco_data.dtypes)
# One-hot encode the categorical variables except 'churn'
columns_to_encode = ['region', 'marital', 'ed', 'retire', 'gender', 'voice', 'internet', 'forward', 'custcat']
telco_data_encoded = pd.get_dummies(telco_data, columns=columns_to_encode, drop_first=True)
# Convert 'churn' to a binary variable where 'Yes' = 1 and 'No' = 0
telco_data_encoded['churn'] = telco_data_encoded['churn'].apply(lambda x: 1 if x == 'Yes' else 0)
# Print the updated DataFrame to confirm changes
print(telco_data_encoded.head())




# STEP 2: FIT SURVIVAL MODELS
# Define the duration and event columns
duration_col = 'tenure'
event_col = 'churn'

# Initialize model fitters
weibull_aft = WeibullAFTFitter()
lognormal_aft = LogNormalAFTFitter()
loglogistic_aft = LogLogisticAFTFitter()

# Fit the models
weibull_aft.fit(telco_data_encoded, duration_col=duration_col, event_col=event_col)
lognormal_aft.fit(telco_data_encoded, duration_col=duration_col, event_col=event_col)
loglogistic_aft.fit(telco_data_encoded, duration_col=duration_col, event_col=event_col)




# STEP 3: PRINT MODEL SUMMARIES
weibull_aft.print_summary()
lognormal_aft.print_summary()
loglogistic_aft.print_summary()




#STEO 4 PLOTS
#VISUALIZE SURIVAL CURVES
# Predicting and plotting the survival functions for a sample of individuals
sample = telco_data_encoded.sample(25, random_state=42)  # Random sample of 25 customers for clearer visualization

weibull_sf = weibull_aft.predict_survival_function(sample)
lognormal_sf = lognormal_aft.predict_survival_function(sample)
loglogistic_sf = loglogistic_aft.predict_survival_function(sample)

plt.figure(figsize=(10, 6))
for i, col in enumerate(sample.index):
    plt.plot(weibull_sf.index, weibull_sf[col], linestyle='-', color='blue', alpha=0.5, label='Weibull AFT' if i == 0 else "_nolegend_")
    plt.plot(lognormal_sf.index, lognormal_sf[col], linestyle='--', color='red', alpha=0.5, label='Log-Normal AFT' if i == 0 else "_nolegend_")
    plt.plot(loglogistic_sf.index, loglogistic_sf[col], linestyle=':', color='green', alpha=0.5, label='Log-Logistic AFT' if i == 0 else "_nolegend_")

plt.title('Survival Curves Comparison')
plt.xlabel('Time in Months')
plt.ylabel('Survival Probability')
plt.legend()
plt.show()




#STEP 5 CALCULATE CLV
#Calculate Customer Lifetime Value (CLV)
# Assuming an average monthly revenue per customer
monthly_revenue = 50 

# Using the Weibull model
expected_lifetimes = weibull_aft.predict_expectation(telco_data_encoded)
telco_data_encoded['CLV'] = expected_lifetimes * monthly_revenue
average_clv = telco_data_encoded['CLV'].mean()
print("Average Customer Lifetime Value: ${:.2f}".format(average_clv))


# Using the Log-Normal model 
expected_lifetimes = lognormal_aft.predict_expectation(telco_data_encoded)
telco_data_encoded['CLV'] = expected_lifetimes * monthly_revenue
average_clv = telco_data_encoded['CLV'].mean()
print("Average Customer Lifetime Value: ${:.2f}".format(average_clv))


# Using the Log-Logistic model 
expected_lifetimes = loglogistic_aft.predict_expectation(telco_data_encoded)
telco_data_encoded['CLV'] = expected_lifetimes * monthly_revenue
average_clv = telco_data_encoded['CLV'].mean()
print("Average Customer Lifetime Value: ${:.2f}".format(average_clv))


#WHICH MODEL WOULD I CHOOSE
# I would likely start with the Weibull model due to its simplicity 
# and ease of interpretation unless performance metrics strongly favor a more complex 
# model like the Log-Normal or Log-Logistic. The final choice would balance the need for 
# accuracy with the practicality of implementation and stakeholder communication.













