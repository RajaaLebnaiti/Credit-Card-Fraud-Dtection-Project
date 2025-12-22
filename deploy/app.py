import gradio as gr
import pandas as pd
import joblib



# Charger le modèle avec joblib 
model = joblib.load("C:/Users/hp/Desktop/Credit Card Fraud Detection Project/src/models/final_model_compressed.joblib")

# Seuil métier
threshold = 0.20

# 🔥 Récupérer automatiquement l'ordre exact des features
features = list(model.feature_names_in_)

def predict_fraud(*inputs):
    data = pd.DataFrame([inputs], columns=features)
    proba = model.predict_proba(data)[:, 1][0]
    prediction = int(proba >= threshold)

    return {
        "Prediction": "Fraud 🚨" if prediction else "Not Fraud ✅",
        "Fraud Probability": round(proba, 3)
    }

interface = gr.Interface(
    fn=predict_fraud,
    inputs=[gr.Number(label=f) for f in features],
    outputs="json",
    title="Credit Card Fraud Detection",
    description="Random Forest fraud detection model (threshold = 0.20)"
)

interface.launch()
