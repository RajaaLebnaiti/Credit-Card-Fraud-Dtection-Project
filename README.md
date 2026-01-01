# 🛡️ Détection de Fraude sur Transactions Bancaires

## 📌 Présentation du projet
Ce projet vise à construire un pipeline complet de **détection de fraude** sur des transactions bancaires.  
Il couvre toutes les étapes d’un workflow MLOps , Machine Learning professionnel : de l’ingestion des données jusqu’au déploiement et au monitoring du modèle.

---

## 📂 Données utilisées
- Dataset public : **Credit Card Fraud Detection (Kaggle)**
Link: https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud

---

## 🔍 Processus du projet

Le projet suit les étapes clés suivantes :

### **1. Ingestion & Exploration**
- Récupération et versionement des données  via DVC
- Analyse exploratoire : valeurs manquantes, outliers, distributions, corrélations  
- Sauvegarde des données propres  

### **2. Préparation & Feature Engineering**
- Nettoyage 
- Encodage des variables  
- Sélection des features pertientes 
- Pipeline automatisé  

### **3. Modélisation**
- Baseline + modèles avancés (RandomForest, DecisonTrees, XGBoost)  
- Métriques adaptées aux données déséquilibrées (AUC, F1, Precision, Recall, Confusion Matrix)  

### **4. Tracking des expériences**
- Utilisation de **MLflow** pour suivre :
  - métriques  
  - paramètres  
  - versions du modèle  

### **5. Tests & Qualité**
- Tests unitaires avec Pytest pour chaque script (preprocessing / training)
 

### **6. CI/CD**
- GitHub Actions pour :
  - exécuter les tests unitaires à chaque push  
  - vérifier le bon fonctionnement du code  

### **7. Déploiement**
- Déploiement du modèle et d’une interface **Gradio** sur **HuggingFace Spaces**
Link: RajaaLE/Credit_Card_Fraud_Dtection_Project

### **8. Monitoring **
- Détection de drift via EvidentlyAI 
- Suivi régulier des performances  

### **9. Dockerisation **
- Conteneurisation du projet via Docker
- Link to docker image : https://hub.docker.com/repository/docker/rajaa191/credit-card-fraud-detection/general
---

