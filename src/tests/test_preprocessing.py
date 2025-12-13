import pandas as pd
import pytest
from io import StringIO

# Import the script of preprocessing
from scripts.preprocessing import preprocess_data


# Teste 1: Test rounding values are correct
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


# Teste 2: Test row count unchanged after rounding
def test_row_count_unchaged():
    df = pd.DataFrame({
        "distance_from_home": [1.234, 2.345],
        "distance_from_last_transaction": [3.456, 4.567],
        "ratio_to_median_purchase_price": [0.123, 0.456]
    })
    
    df_out = preprocess_data(df)
    
    assert len(df_out) == len(df)
    

# Teste 3: Test missing column
def test_missing_column():
    df = pd.DataFrame({
        "distance_from_home": [1.23],
        "ratio_to_median_purchase_price": [0.12]
    })
    
    with pytest.raises(KeyError):
        preprocess_data(df)
        

# Teste 4: Test null values
def test_nan_are_perserved():
    df = pd.DataFrame({
        "distance_from_home": [1.234, None],
        "distance_from_last_transaction": [3.456, 5.678],
        "ratio_to_median_purchase_price": [0.123, None]
    })
    df_out = preprocess_data(df)
    
    assert pd.isna(df_out['distance_from_home'].iloc[1])
    assert pd.isna(df_out['ratio_to_median_purchase_price'].iloc[1])