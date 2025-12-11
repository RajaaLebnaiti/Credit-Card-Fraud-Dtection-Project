import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # remonte de scripts/ to src/
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "card_transdata.csv")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessing", "clean_data_card_transdata.csv")


# Function to round the values to only 2 decimals
def preprocess_data(df):
    df['distance_from_home'] = df['distance_from_home'].round(2)
    df['distance_from_last_transaction'] = df['distance_from_last_transaction'].round(2)
    df['ratio_to_median_purchase_price']= df['ratio_to_median_purchase_price'].round(2)
    return df



# Load raw dataset
df = pd.read_csv(DATA_PATH)

# Appky preprocessing function
df = preprocess_data(df)

# Save the data
df.to_csv(CLEAN_DATA_PATH, index= False)
print("The data has been saved successfully ! ")