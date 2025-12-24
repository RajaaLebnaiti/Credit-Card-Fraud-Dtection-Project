import pandas as pd
import numpy as np
from io import StringIO
from scripts.training import split_data_features_target, split_data_train_test, RandomForestModel, predict_with_treshold, compute_metrics, run_training_pipeline
import pytest
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Test 1: testing the function of spliting data into features and target
def test_split_data_features_target():
    # syntethic data
    csv = StringIO("distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order,fraud\n"
                    "57.88,0.31,1.95,1.0,1.0,0.0,0.0,0.0\n"
                    "10.83,0.18,1.29,1.0,0.0,0.0,0.0,1.0\n"
                    "5.09,0.81,0.43,1.0,0.0,0.0,1.0,0.0")
    df = pd.read_csv(csv)
    X, y = split_data_features_target(df)
    
    assert X.shape == (3,6), "The shape of X should be (3, 6)"
    assert y.shape == (3,), "The shape should be (3,)"
    assert "fraud" not in X.columns
    assert "ratio_to_median_purchase_price" not in X.columns






# Test 2: testing the function of splitting data into train and test
def test_split_data_train_test():
    csv = StringIO("distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order,fraud\n"
                    "57.88,0.31,1.95,1.0,1.0,0.0,0.0,0.0\n"
                    "10.83,0.18,1.29,1.0,0.0,0.0,0.0,1.0\n"
                    "5.09,0.81,0.43,1.0,0.0,0.0,1.0,0.0")
    df = pd.read_csv(csv)
    X_train, X_test, y_train, y_test = split_data_train_test(df)
    
    assert len (X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) == 2
    assert len(X_test) == 1
    


# Test 3: Testing the function of training the model

def test_RandomForestModel():
    csv = StringIO("distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order,fraud\n"
                    "57.88,0.31,1.95,1.0,1.0,0.0,0.0,0.0\n"
                    "10.83,0.18,1.29,1.0,0.0,0.0,0.0,1.0\n"
                    "5.09,0.81,0.43,1.0,0.0,0.0,1.0,0.0")
    df = pd.read_csv(csv)
    X_train, X_test, y_train, y_test = split_data_train_test(df)
    params = {
    'n_estimators': 200,
    'criterion': 'gini',
    'random_state': 42
    }
    model_1 = RandomForestModel(params, X_train, y_train)
    model_2 = RandomForestModel(params, X_train, y_train)
    
    assert len(model_1.estimators_) == len(model_2.estimators_) == 200, "Incorrect number of estimators"





  
# Test 4: Testing the function to make predictions with treshold

def test_predict_with_treshold():
    csv = StringIO("distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order,fraud\n"
                    "57.88,0.31,1.95,1.0,1.0,0.0,0.0,0.0\n"
                    "10.83,0.18,1.29,1.0,0.0,0.0,0.0,1.0\n"
                    "5.09,0.81,0.43,1.0,0.0,0.0,1.0,0.0")

    df = pd.read_csv(csv)
    X_train, X_test, y_train, y_test = split_data_train_test(df)
    params = {
    'n_estimators': 200,
    'criterion': 'gini',
    'random_state': 42
    }
    model_1 = RandomForestModel(params, X_train, y_train)
    model_2 = RandomForestModel(params, X_train, y_train)
    

    
    y_proba_1 = predict_with_treshold(model_1, X_test)
    y_proba_2 = predict_with_treshold(model_2, X_test)

    assert np.array_equal(y_proba_1, y_proba_2), "Model predictions are not reproducible with a fixed seed"
    
    assert y_proba_1.shape == (len(X_test),)
    assert set(np.unique(y_proba_1)).issubset({0,1})


# Test 5: Testing the function for computing metrics    
def test_compute_metrics():
    csv = StringIO("distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order,fraud\n"
                    "57.88,0.31,1.95,1.0,1.0,0.0,0.0,0.0\n"
                    "10.83,0.18,1.29,1.0,0.0,0.0,0.0,1.0\n"
                    "5.09,0.81,0.43,1.0,0.0,0.0,1.0,0.0")
    
    df = pd.read_csv(csv)
    X_train, X_test, y_train, y_test = split_data_train_test(df)
    params = {
    'n_estimators': 200,
    'criterion': 'gini',
    'random_state': 42
    }
    model = RandomForestModel(params, X_train, y_train)
    y_proba = predict_with_treshold(model, X_test)
    
    metrics = compute_metrics(y_test, y_proba)
    
    
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    
    assert metrics["precision"] == precision_score(y_test, y_proba)
    assert metrics["recall"] == recall_score(y_test, y_proba)
    assert metrics["f1"] == f1_score(y_test, y_proba)
    
    
    
    
  
# Test 6: Testing the General Pipeline
def test_run_training_pipeline():
    
    csv = StringIO("distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order,fraud\n"
                    "57.88,0.31,1.95,1.0,1.0,0.0,0.0,0.0\n"
                    "10.83,0.18,1.29,1.0,0.0,0.0,0.0,1.0\n"
                    "5.09,0.81,0.43,1.0,0.0,0.0,1.0,0.0")
    
    df = pd.read_csv(csv)
    
    params = {
    'n_estimators': 200,
    'criterion': 'gini',
    'random_state': 42
    }
    model, metrics = run_training_pipeline(df, params)
    
    assert isinstance(model, RandomForestClassifier)
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {"precision", "recall", "f1"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
