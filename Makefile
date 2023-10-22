# Makefile for Apprentissage Automatique Project

# Default target
all: help

# Update project dependencies
requirements:
	pip install --upgrade -r requirements.txt

# Random forest pour la prédiction du retard (partie 1)
retard:
	python -m src.Train.train_random_forest1

# Random forest pour la prédiction des causes (partie 2)
causes:
	python -m src.Train.train_random_forest2

# Display help message
help:
	@echo "Apprentissage Automatique Project Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  requirement  - Install project libaries"
	@echo "  retard       - Train the random forest model for the delay and plot the prediction result"
	@echo "  causes       - Train the random forest model fro the causes and plot the prediction result"
	@echo "  help         - Display this help message"
