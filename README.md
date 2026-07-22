# Disaster Response Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_App-lightgrey?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Pipeline-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning pipeline that displays 36 emergency categories and trains classifiers for the 35 categories with both positive and negative examples. The project includes a validated ETL pipeline, a multi-output text classifier, generated evaluation evidence, and a Flask web application for local message classification with interactive visualizations.

## Motivation

During disasters, emergency response organizations receive thousands of messages via social media, news, and direct channels. Manually sorting these messages to route them to the correct relief agency is slow and error-prone. This project automates that classification using NLP and machine learning, enabling faster response to people in need.

## Project Structure

```
Disaster-Response-Pipeline/
├── app/
│   ├── run.py                          # Flask web application
│   └── templates/
│       ├── master.html                 # Main page template
│       └── go.html                     # Classification results template
├── data/
│   ├── disaster_messages.csv           # Raw messages dataset (26,248 rows)
│   ├── disaster_categories.csv         # Raw categories dataset (26,248 rows)
│   ├── etl_metrics.json                 # Generated data-quality evidence
│   └── process_data.py                 # ETL pipeline script
├── models/
│   ├── model_metrics.json               # Generated holdout evidence
│   └── train_classifier.py              # ML pipeline script
├── scripts/
│   └── check_runtime.py                # Database/model/route smoke check
├── tests/
│   ├── test_process_data.py            # ETL pipeline tests
│   ├── test_train_classifier.py        # Tokenizer and model tests
│   └── test_app.py                     # Saved-artifact and Flask route tests
├── notebooks/
│   ├── ETL Pipeline Preparation.ipynb  # ETL development notebook
│   └── ML Pipeline Preparation.ipynb   # ML development notebook
├── utils.py                            # Shared tokenizer module
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Multi-stage Docker build
├── docker-compose.yml                  # One-command Docker setup
├── .gitignore
├── LICENSE                             # MIT License
└── README.md
```

## How It Works

The project follows a three-stage pipeline:

### 1. ETL Pipeline (`data/process_data.py`)
- Loads messages and categories from CSV files
- Validates source schemas and requires matching message/category IDs
- Prevents repeated source IDs from creating a many-to-many join
- Splits categories into 36 binary columns and unions conflicting duplicate annotations
- Emits one row per message ID and excludes six blank or spreadsheet-error messages
- Stores cleaned data in a SQLite database

### 2. ML Pipeline (`models/train_classifier.py`)
- Loads cleaned data from the SQLite database
- Tokenizes with deterministic regex parsing, bundled English stop words, and Snowball stemming
- Requires no runtime corpus downloads, so training and inference use the same transform offline
- Drops single-class categories (e.g. `child_alone` with zero positive examples)
- Builds a multi-output classification pipeline: `CountVectorizer` → `TF-IDF` → `SGDClassifier` (logistic regression)
- Uses `class_weight='balanced'` to handle severe class imbalance in rare categories
- Uses iterative multilabel stratification for the holdout and cross-validation folds
- Optimizes regularization strength with `GridSearchCV` (scored on weighted F1)
- Saves category order and library metadata with the model, plus machine-readable holdout metrics

### 3. Web Application (`app/run.py`)
- Flask app with 4 interactive Plotly visualizations:
  - Distribution of Message Genres
  - Top 10 Message Categories
  - Category Correlation Heatmap
  - Average Message Length by Genre
- Real-time message classification interface

## Installation

### Prerequisites
- Python 3.11 (or Docker)
- [uv](https://docs.astral.sh/uv/) is recommended; pip also works

### Setup

1. Clone the repository:
```bash
git clone https://github.com/tobibolu/Disaster-Response-Pipeline.git
cd Disaster-Response-Pipeline
```

2. Create the pinned environment and install dependencies:
```bash
uv venv --python 3.11 .venv
uv pip install -r requirements.txt --python .venv/bin/python
source .venv/bin/activate
```

3. Run the ETL pipeline to process data and create the database:
```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

4. Run the ML pipeline to train the classifier:
```bash
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

The training command also writes `models/model_metrics.json`. On the verified local run it completed in under one minute; timing depends on hardware.

5. Verify the full runtime without starting a server:
```bash
python scripts/check_runtime.py
```

6. Start the web application:
```bash
python app/run.py
```

7. Open your browser and go to `http://localhost:3001/`. Stop the server with `Control-C`.

### Docker (Alternative)

Run the entire pipeline and web app with one command:

```bash
docker-compose up --build
```

This builds the database, trains the model with the pinned dependency matrix, and starts the Flask server. Open `http://localhost:3001/` when the build completes.

> **Note:** The initial Docker build takes 5-15 minutes due to model training. Subsequent runs use cached layers.

## The 36 Categories

Messages are classified across these emergency response categories:

| Category | Category | Category | Category |
|---|---|---|---|
| Related | Request | Offer | Aid Related |
| Medical Help | Medical Products | Search And Rescue | Security |
| Military | Child Alone* | Water | Food |
| Shelter | Clothing | Money | Missing People |
| Refugees | Death | Other Aid | Infrastructure Related |
| Transport | Buildings | Electricity | Tools |
| Hospitals | Shops | Aid Centers | Other Infrastructure |
| Weather Related | Floods | Storm | Fire |
| Earthquake | Cold | Other Weather | Direct Report |

*\*`child_alone` has zero positive examples in the dataset and is excluded from model training.*

## Model performance

The 22 July 2026 remediation run trained on 20,939 messages and evaluated once on a multilabel-stratified holdout of 5,235 messages. Its weighted F1 is 0.661, but macro F1 is only 0.436. That gap matters: frequent labels perform much better than rare categories, where the recall-forward `class_weight='balanced'` policy creates many false positives.

| Category | Precision | Recall | F1-Score |
|---|---|---|---|
| Earthquake | 0.81 | 0.87 | 0.84 |
| Weather Related | 0.76 | 0.81 | 0.78 |
| Food | 0.69 | 0.86 | 0.77 |
| Storm | 0.60 | 0.87 | 0.71 |
| Water | 0.56 | 0.89 | 0.69 |
| Shelter | 0.55 | 0.82 | 0.66 |
| Request | 0.58 | 0.77 | 0.66 |
| Direct Report | 0.52 | 0.74 | 0.61 |
| **Weighted average** | **0.62** | **0.75** | **0.66** |
| **Macro average** | **0.35** | **0.63** | **0.44** |

The generated evidence also records micro F1 0.604, exact-match accuracy 0.176, and Hamming loss 0.089. These are offline dataset metrics, not evidence of production routing performance. The model uses TF-IDF features, which are strongest on explicit wording and weaker on indirect meaning.

## Dataset

The source files are provided by [Appen](https://appen.com/) (formerly Figure Eight) and contain real disaster-response records used in the Udacity Data Science Nanodegree project. The raw files each contain 26,248 rows but only 26,180 distinct IDs. Thirty-six IDs have multiple distinct category strings; the ETL retains the union of positive annotations rather than duplicating the message. After that resolution and the exclusion of two blank messages plus four `#NAME?` spreadsheet-error messages, the modelling table contains 26,174 unique messages. The exact counts and policy are generated in `data/etl_metrics.json`.

- **Messages:** Real messages sent during disaster events
- **Genres:** Direct messages, news articles, and social media posts
- **Categories:** Multi-label binary classification (a message can belong to multiple categories)

## Tech Stack

- **Python** - Core language
- **pandas / NumPy** - Data manipulation
- **scikit-learn** - Machine learning pipeline (SGDClassifier, GridSearchCV, MultiOutputClassifier)
- **NLTK / scikit-learn** - Offline stemming and bundled English stop words
- **Flask** - Web application framework
- **Plotly** - Interactive data visualizations
- **SQLAlchemy** - Database ORM
- **SQLite** - Data storage

## Running Tests

```bash
python -m pytest tests/ -v
python scripts/check_runtime.py
```

The current local verification result is **22 tests passed**, followed by a successful database/model/prediction-route smoke check. Repeated evaluation, uncertainty intervals, threshold tuning, and error slices remain future evaluation work rather than completed claims.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Appen](https://appen.com/) for providing the disaster response dataset
