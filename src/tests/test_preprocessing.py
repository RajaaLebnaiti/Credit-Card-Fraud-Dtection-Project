import pandas as pd
import pytest
from io import StringIO

# Import the script of preprocessing
from scripts.preprocessing import preprocess_data


def test_rounding_values():
    # Fake input as csv
    csv = StringIO("distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,fraud\n"
        "10.12345,20.98765,1.23456,0\n"
        "5.54321,3.14159,2.71828,1\n")
    
    df = pd.read_csv(csv)
    df_out = preprocess_data(df)
    
    assert df_out['distance_from_home'].iloc[0] == 10.12
    assert df_out['distance_from_last_transaction'].iloc[0] == 20.99
    assert df_out['ratio_to_median_purchase_price'].iloc[0] == 1.23
    
    assert df_out['distance_from_home'].iloc[1] == 5.54
    assert df_out['distance_from_last_transaction'].iloc[1] == 3.14
    assert df_out['ratio_to_median_purchase_price'].iloc[1] == 2.72
