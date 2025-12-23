import pandas as pd
from io import StringIO
from scripts.training import split_data_features_target, split_data_train_test, RandomForestClassifier, predict_with_treshold, compute_metrics, run_training_pipeline
import pytest


# Test 1: testing the function of spliting data into features and target
def test_split_data_features_target():
    # syntethic data
    csv = StringIO("distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order,fraud\n"
                    "57.88,0.31,1.95,1.0,1.0,0.0,0.0,0.0\n"
                    "10.83,0.18,1.29,1.0,0.0,0.0,0.0,0.0\n"
                    "5.09,0.81,0.43,1.0,0.0,0.0,1.0,0.0")
    df = pd.read_csv(csv)
    X, y = split_data_features_target(df)
    
    assert X.shape == (3,6), "The shape of X should be (3, 6)"
    assert y.shape == (3,), "The shape should be (3,)"
    assert "fraud", "ratio_to_median_purchase_price" not in X.columns


# Test 2: testing the function of splitting data into train and test
def test_split_data_train_test():
    csv = StringIO("distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order,fraud\n"
                    "57.88,0.31,1.95,1.0,1.0,0.0,0.0,0.0\n"
                    "10.83,0.18,1.29,1.0,0.0,0.0,0.0,0.0\n"
                    "5.09,0.81,0.43,1.0,0.0,0.0,1.0,0.0")
    df = pd.read_csv(csv)
    X_train, X_test, y_train, y_test = split_data_train_test(df)
    
    assert len (X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) == 2
    assert len(X_test) == 1
    


# Test 3: Testing the function of training the model
