import pandas as pd
import os

RAW_DATA_PATH = "src/data/raw/card_transdata.csv"
CLEAN_DATA_PATH = "src/data/preprocessing/clean_data_card_transdata.csv"


# Function to round the values to only 2 decimals
def preprocess_data(df):
    df['distance_from_home'] = df['distance_from_home'].round(2)
    df['distance_from_last_transaction'] = df['distance_from_last_transaction'].round(2)
    df['ratio_to_median_purchase_price']= df['ratio_to_median_purchase_price'].round(2)
    return df



def main():
    df = pd.read_csv(RAW_DATA_PATH)
    df_clean = preprocess_data(df)

    df_clean.to_csv(CLEAN_DATA_PATH, index=False)


if __name__ == "__main__":
    main()
