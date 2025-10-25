# Utiliser une image de base Python légère
FROM python:3.11-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier de dépendances et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers du projet (script, données et modèle)
# Le fichier ai4i2020.csv doit être dans le même répertoire que le Dockerfile
COPY mlops_project_script.py .
COPY ai4i2020.csv .
# Le modèle sera créé lors de la première exécution du script
# Pour une exécution complète dans le conteneur, nous lançons le script

# Définir la commande par défaut pour exécuter le script
CMD ["python", "mlops_project_script.py"]
