import pandas as pd

# Function to round the values to only 2 decimals
def preprocess_data(df):
    df['distance_from_home'] = df['distance_from_home'].round(2)
    df['distance_from_last_transaction'] = df['distance_from_last_transaction'].round(2)
    df['ratio_to_median_purchase_price']= df['ratio_to_median_purchase_price'].round(2)
    return df


