# Modular-ML-Pipeline

# Modular Machine Learning Pipeline – Heart Disease Dataset 🚢

This project is a **modular machine learning pipeline** built on the Titanic dataset.  
It follows a clean, structured, and reusable approach to building ML projects.  

The pipeline is divided into the following modules:
- **Data Preprocessing** 🧹
- **Feature Engineering** 🔧
- **Model Training** 🤖
- **Model Evaluation** 📊

---

## 📂 Project Structure

├── data/
│ ├── raw/ # Original dataset files (input)
│ ├── processed/ # Cleaned and processed datasets (output)
│
├── preprocessing.py # Data loading & preprocessing steps
├── config.py # Feature transformation & encoding
├── train_model.py # Model training script
├── evaluate.py # Model evaluation script
├── utils.py # Helper functions
├── main.py # Orchestrator (runs full pipeline)
├── requirements.txt # Dependencies
└── README.md # Project documentation

Installed Required Dependies:
pip install -r requirements.txt


Dataset

Place your dataset inside the data/raw/ directory.

For the Titanic dataset, download from Kaggle Titanic Dataset
.

Example file:
data/raw/train.csv
data/raw/test.csv

▶️ Usage

You can run the full pipeline with:
python main.py 

This will automatically:

Load and preprocess the raw dataset

Apply feature engineering

Train the ML model (e.g., Logistic Regression, Random Forest, etc.)

Evaluate the model and print metrics

🧩 Run Modules Individually

You can also run specific pipeline stages for debugging:
python preprocessing.py       # Preprocess dataset
python utils.py # Feature transformations
python train_model.py               # Train the model
python evaluate.py            # Evaluate trained model









