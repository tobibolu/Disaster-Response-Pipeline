# Disaster Response Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_App-lightgrey?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Pipeline-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning pipeline that classifies disaster response messages into 36 emergency categories. The project includes an ETL pipeline for data processing, an ML pipeline for multi-output text classification, and a Flask web application for real-time message classification with interactive visualizations.

## Motivation

During disasters, emergency response organizations receive thousands of messages via social media, news, and direct channels. Manually sorting these messages to route them to the correct relief agency is slow and error-prone. This project automates that classification using NLP and machine learning, enabling faster response to people in need.

## Screenshots

> **Web App Homepage** - Interactive visualizations showing message genre distribution, top categories, category correlations, and message length analysis.

> **Classification Results** - Enter a disaster message and see it classified across 36 emergency categories in real time.

*To capture screenshots, run the web app locally and navigate to `http://localhost:3001/`.*

## Project Structure

```
Disaster-Response-Pipeline/
├── app/
│   ├── run.py                          # Flask web application
│   └── templates/
│       ├── master.html                 # Main page template
│       └── go.html                     # Classification results template
├── data/
│   ├── disaster_messages.csv           # Raw messages dataset (26,249 rows)
│   ├── disaster_categories.csv         # Raw categories dataset (26,249 rows)
│   └── process_data.py                 # ETL pipeline script
├── models/
│   └── train_classifier.py             # ML pipeline script
├── tests/
│   ├── test_process_data.py            # ETL pipeline tests
│   └── test_train_classifier.py        # ML pipeline tests
├── notebooks/
│   ├── ETL Pipeline Preparation.ipynb  # ETL development notebook
│   └── ML Pipeline Preparation.ipynb   # ML development notebook
├── utils.py                            # Shared tokenizer module
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
└── README.md
```

## How It Works

The project follows a three-stage pipeline:

### 1. ETL Pipeline (`data/process_data.py`)
- Loads messages and categories from CSV files
- Merges datasets and splits categories into 36 binary columns
- Cleans data: removes duplicates, clips non-binary values to 0/1
- Stores cleaned data in a SQLite database

### 2. ML Pipeline (`models/train_classifier.py`)
- Loads cleaned data from the SQLite database
- Tokenizes text using a shared NLP pipeline (lowercase, remove punctuation, remove stopwords, lemmatize)
- Builds a multi-output classification pipeline: `CountVectorizer` → `TF-IDF` → `RandomForestClassifier`
- Optimizes hyperparameters with `GridSearchCV` (scored on weighted F1)
- Evaluates on test data and saves the trained model

### 3. Web Application (`app/run.py`)
- Flask app with 4 interactive Plotly visualizations:
  - Distribution of Message Genres
  - Top 10 Message Categories
  - Category Correlation Heatmap
  - Average Message Length by Genre
- Real-time message classification interface

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/tobibolu/Disaster-Response-Pipeline.git
cd Disaster-Response-Pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the ETL pipeline to process data and create the database:
```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

4. Run the ML pipeline to train the classifier:
```bash
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

> **Note:** Training may take 15-30 minutes depending on your hardware.

5. Start the web application:
```bash
python app/run.py
```

6. Open your browser and go to `http://localhost:3001/`

## The 36 Categories

Messages are classified across these emergency response categories:

| Category | Category | Category | Category |
|---|---|---|---|
| Related | Request | Offer | Aid Related |
| Medical Help | Medical Products | Search And Rescue | Security |
| Military | Child Alone | Water | Food |
| Shelter | Clothing | Money | Missing People |
| Refugees | Death | Other Aid | Infrastructure Related |
| Transport | Buildings | Electricity | Tools |
| Hospitals | Shops | Aid Centers | Other Infrastructure |
| Weather Related | Floods | Storm | Fire |
| Earthquake | Cold | Other Weather | Direct Report |

## Dataset

The dataset is provided by [Appen](https://appen.com/) (formerly Figure Eight) and contains 26,249 real disaster response messages collected from multiple sources. Each message is labeled across 36 categories.

- **Messages:** Real messages sent during disaster events
- **Genres:** Direct messages, news articles, and social media posts
- **Categories:** Multi-label binary classification (a message can belong to multiple categories)

## Tech Stack

- **Python** - Core language
- **pandas / NumPy** - Data manipulation
- **scikit-learn** - Machine learning pipeline, GridSearchCV, classification
- **NLTK** - Natural language processing (tokenization, lemmatization, stopwords)
- **Flask** - Web application framework
- **Plotly** - Interactive data visualizations
- **SQLAlchemy** - Database ORM
- **SQLite** - Data storage

## Running Tests

```bash
python -m pytest tests/ -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Udacity](https://www.udacity.com/) Data Science Nanodegree for the project framework
- [Appen](https://appen.com/) for providing the disaster response dataset
