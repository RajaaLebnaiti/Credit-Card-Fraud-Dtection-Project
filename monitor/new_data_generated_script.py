# Because we don't have access to new data, this script is used to add some noise on the 
# data to generate current_data used in evidently AI for monitoring.

import pandas as pd
import numpy as np

# Load data
initial_data = pd.read_csv('C:/Users/hp/Desktop/Credit Card Fraud Detection Project/src/data/preprocessing/clean_data_card_transdata.csv')

# Path to save generated data
GENRATED_DATA_PATH = 'C:/Users/hp/Desktop/Credit Card Fraud Detection Project/monitor/generated_data.csv'

# Split to features and target
X = initial_data.drop(columns=['fraud', 'ratio_to_median_purchase_price'])    
y = initial_data['fraud']


# Then we add some noise to the data
gaussian_noise = np.random.normal(0, 2, len(X)) # We use other distributions, 
X[['distance_from_home','distance_from_last_transaction']] = X[['distance_from_home','distance_from_last_transaction']].add(gaussian_noise, axis=0)

# save current_data
current_data =  X.copy()
current_data['fraud'] = y

current_data.to_csv(GENRATED_DATA_PATH, index=False)
print("The generated data has been saved successfully!")