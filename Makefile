# Makefile for AapprAuto Project

# Default target
all: help

# Update project dependencies
requirement:
	pip install --upgrade -r requirements.txt

# Train the machine learning model
train:
	python src/train.py

# Evaluate the machine learning model
evaluate:
	python src/evaluate.py


# Display help message
help:
	@echo "Machine Learning Project Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  venv      - Create a virtual environment"
	@echo "  install   - Install project dependencies"
	@echo "  update    - Update project dependencies"
	@echo "  train     - Train the machine learning model"
	@echo "  evaluate  - Evaluate the machine learning model"
	@echo "  clean     - Clean up generated files and build artifacts"
	@echo "  help      - Display this help message"

.PHONY: venv install update train evaluate clean help