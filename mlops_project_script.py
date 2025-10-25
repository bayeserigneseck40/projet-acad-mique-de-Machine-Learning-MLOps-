{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import accuracy_score, classification_report\n",
        "import joblib\n",
        "import os"
      ],
      "metadata": {
        "id": "UZLUguTmMeGQ"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "\n",
        "# --- Étape 1 : Acquisition et Préparation des Données (Data Engineering) ---\n",
        "\n",
        "def load_and_prepare_data(file_path):\n",
        "    \"\"\"Charge les données, nettoie et effectue le Feature Engineering.\"\"\"\n",
        "    print(\"1. Chargement et préparation des données...\")\n",
        "\n",
        "    # Chargement des données\n",
        "    df = pd.read_csv(file_path)\n",
        "\n",
        "    # Renommage des colonnes (pour gérer le BOM et les espaces)\n",
        "    df.columns = ['UDI', 'Product_ID', 'Type', 'Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min', 'Machine_failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']\n",
        "\n",
        "    # La colonne 'Machine_failure' est notre cible\n",
        "    df.rename(columns={'Machine_failure': 'Failure'}, inplace=True)\n",
        "\n",
        "    # Feature Engineering simple : Calcul de la différence de température\n",
        "    df['Temp_Diff'] = df['Process_temperature_K'] - df['Air_temperature_K']\n",
        "\n",
        "    # Suppression des colonnes non nécessaires et des colonnes de défaillance individuelles\n",
        "    failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']\n",
        "    df = df.drop(columns=['UDI', 'Product_ID'] + failure_cols)\n",
        "\n",
        "    # Séparation des features (X) et de la cible (y)\n",
        "    X = df.drop('Failure', axis=1)\n",
        "    y = df['Failure']\n",
        "\n",
        "    print(f\"   - Taille du jeu de données après nettoyage : {X.shape}\")\n",
        "    print(f\"   - Nombre de défaillances (cible=1) : {y.sum()}\")\n",
        "\n",
        "    return X, y\n"
      ],
      "metadata": {
        "id": "E_3KD8kHMhkX"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "# --- Étape 2 : Modélisation Machine Learning (Data Science) ---\n",
        "\n",
        "def train_and_evaluate_model(X, y):\n",
        "    \"\"\"Entraîne un modèle de classification et évalue ses performances.\"\"\"\n",
        "    print(\"\\n2. Entraînement et évaluation du modèle...\")\n",
        "\n",
        "    # Séparation des données en ensembles d'entraînement et de test\n",
        "    # Stratify=y est crucial car les défaillances sont rares (classe déséquilibrée)\n",
        "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
        "\n",
        "    # Définition des colonnes à transformer\n",
        "    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns\n",
        "    categorical_features = X.select_dtypes(include=['object']).columns\n",
        "\n",
        "    # Création du préprocesseur (StandardScaler pour les numériques, OneHotEncoder pour les catégorielles)\n",
        "    preprocessor = ColumnTransformer(\n",
        "        transformers=[\n",
        "            ('num', StandardScaler(), numerical_features),\n",
        "            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)\n",
        "        ],\n",
        "        remainder='passthrough'\n",
        "    )\n",
        "\n",
        "    # Création du pipeline (Préprocesseur + Modèle)\n",
        "    model_pipeline = Pipeline(steps=[\n",
        "        ('preprocessor', preprocessor),\n",
        "        # class_weight='balanced' pour gérer le déséquilibre des classes\n",
        "        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))\n",
        "    ])\n",
        "\n",
        "    # Entraînement du modèle\n",
        "    model_pipeline.fit(X_train, y_train)\n",
        "\n",
        "    # Prédiction sur l'ensemble de test\n",
        "    y_pred = model_pipeline.predict(X_test)\n",
        "\n",
        "    # Évaluation\n",
        "    accuracy = accuracy_score(y_test, y_pred)\n",
        "    report = classification_report(y_test, y_pred, target_names=['No Failure', 'Failure'])\n",
        "\n",
        "    print(f\"   - Précision (Accuracy) du modèle : {accuracy:.4f}\")\n",
        "    print(\"\\n   - Rapport de classification :\\n\", report)\n",
        "\n",
        "    # Sauvegarde du modèle entraîné (pour l'étape MLOps)\n",
        "    model_filename = 'predictive_maintenance_model.joblib'\n",
        "    joblib.dump(model_pipeline, model_filename)\n",
        "    print(f\"\\n3. Modèle sauvegardé sous : {model_filename}\")\n",
        "\n",
        "    return model_pipeline"
      ],
      "metadata": {
        "id": "xRwR2_GkMmGy"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ox58dz7aMap4",
        "outputId": "4811204a-9583-4c26-ee20-1e6dec802351"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1. Chargement et préparation des données...\n",
            "   - Taille du jeu de données après nettoyage : (10000, 7)\n",
            "   - Nombre de défaillances (cible=1) : 339\n",
            "\n",
            "2. Entraînement et évaluation du modèle...\n",
            "   - Précision (Accuracy) du modèle : 0.9860\n",
            "\n",
            "   - Rapport de classification :\n",
            "               precision    recall  f1-score   support\n",
            "\n",
            "  No Failure       0.99      1.00      0.99      1932\n",
            "     Failure       0.93      0.63      0.75        68\n",
            "\n",
            "    accuracy                           0.99      2000\n",
            "   macro avg       0.96      0.82      0.87      2000\n",
            "weighted avg       0.99      0.99      0.98      2000\n",
            "\n",
            "\n",
            "3. Modèle sauvegardé sous : predictive_maintenance_model.joblib\n",
            "\n",
            "--- Pipeline d'Entraînement Terminé ---\n"
          ]
        }
      ],
      "source": [
        "\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    DATA_FILE = 'ai4i2020.csv'\n",
        "\n",
        "    if not os.path.exists(DATA_FILE):\n",
        "        print(f\"Erreur : Le fichier de données '{DATA_FILE}' est introuvable.\")\n",
        "        print(\"Veuillez vous assurer qu'il est dans le répertoire courant.\")\n",
        "    else:\n",
        "        # Étape 1\n",
        "        X, y = load_and_prepare_data(DATA_FILE)\n",
        "\n",
        "        # Étape 2\n",
        "        trained_model = train_and_evaluate_model(X, y)\n",
        "\n",
        "        print(\"\\n--- Pipeline d'Entraînement Terminé ---\")\n"
      ]
    }
  ]
}