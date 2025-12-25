import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

import os
import joblib

# Function to split the data into features and target 
def split_data_features_target(df):
    X = df.drop(columns=['fraud', 'ratio_to_median_purchase_price'])    
    y = df['fraud']
    return X, y



# Function to split the data into training and testing
def split_data_train_test(df, test_size= 0.2, random_state= 42):
    X, y = split_data_features_target(df)
    return train_test_split(X, y, test_size= test_size, random_state= random_state)


# Function of training the model
params = {
    'n_estimators': 200,
    'criterion': 'gini',
    'random_state': 42
}

def RandomForestModel(params, X_train, y_train):
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


# Function to make predictions with treshold
def predict_with_treshold(model, X_test, treshold= 0.2):
    y_proba = model.predict_proba(X_test)[:, 1]
    return (y_proba >= treshold).astype(int)
  

# Function to calculate metrics
def compute_metrics(y_test, y_pred_t):
    return{
        "precision": precision_score(y_test, y_pred_t),
        "recall": recall_score(y_test, y_pred_t),
        "f1": f1_score(y_test, y_pred_t)
    }    


# Genral Pipeline

def run_training_pipeline(df, params, treshold= 0.2):
    X_train, X_test, y_train, y_test = split_data_train_test(df)
    model = RandomForestModel(params, X_train, y_train)
    
    y_pred_t = predict_with_treshold(model, X_test, treshold)
    
    metrics = compute_metrics(y_test, y_pred_t)
    
    return model, metrics






DATA_PATH = "src/data/preprocessing/clean_data_card_transdata.csv"
MODEL_PATH = "src/models/final_model_compressed.joblib"
METRICS_PATH = "src/models/metrics/metrics.csv"



def main():
    params = {
        "n_estimators": 200,
        "criterion": "gini",
        "random_state": 42
    }

    df = pd.read_csv(DATA_PATH)

    model, metrics = run_training_pipeline(df, params, treshold= 0.2)
    
    
    # Create directories to save model and metrics 
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok= True)
    
    # To save the final model
    joblib.dump(model, MODEL_PATH)

    # To save the metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(METRICS_PATH, index=False)

if __name__ == "__main__":
    main()









