# 🛡️ Détection de Fraude sur Transactions Bancaires

## 📌 Présentation du projet
Ce projet vise à construire un pipeline complet de **détection de fraude** sur des transactions bancaires.  
Il couvre toutes les étapes d’un workflow Machine Learning professionnel : de l’ingestion des données jusqu’au déploiement et au monitoring du modèle.

---

## 📂 Données utilisées
- Dataset public : **Credit Card Fraud Detection (Kaggle)**
- Sources complémentaires possibles :
  - APIs **Open Banking** (transactions simulées)
  - Génération synthétique (**Faker**)

---

## 🔍 Processus du projet

Le projet suit les étapes clés suivantes :

### **1. Ingestion & Exploration**
- Récupération et structuration des données  
- Analyse exploratoire : valeurs manquantes, outliers, distributions, corrélations  
- Sauvegarde des données propres  

### **2. Préparation & Feature Engineering**
- Nettoyage, imputation, normalisation  
- Encodage des variables  
- Gestion du déséquilibre (SMOTE / poids de classes)  
- Création et sélection de nouvelles features  
- Pipeline automatisé  

### **3. Modélisation**
- Baseline + modèles avancés (ex : XGBoost, LightGBM)  
- Validation croisée  
- Optimisation des hyperparamètres  
- Métriques adaptées aux données déséquilibrées (AUC, F1)  

### **4. Tracking des expériences**
- Utilisation de **MLflow** pour suivre :
  - métriques  
  - paramètres  
  - artefacts  
  - versions du modèle  

### **5. Tests & Qualité**
- Tests unitaires (nettoyage, encodage, pipeline)  
- Tests d’intégration (pipeline complet)  

### **6. CI/CD**
- GitHub Actions pour :
  - exécuter les tests  
  - vérifier la qualité du code  
  - préparer les modèles  

### **7. Déploiement**
- Déploiement du modèle et d’une interface **Gradio** sur **HuggingFace Spaces**

### **8. Monitoring **
- Détection de drift  
- Suivi régulier des performances  

---

