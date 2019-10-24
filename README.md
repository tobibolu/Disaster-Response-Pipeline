# Disaster Response Pipeline Project

## Project Summary:
    This goal of this project is to create a machine learning pipeline that categorises emergency messages based on the needs communicated by the sender.

### Files:
1. process_data.py - Processes message & category data from CSV files and loans the data into a sqlite database

2. train_classifier.py - Reads from the sqlite database to create and save a multi-output supervised learning model

3. run.py - Web app that extracts data from the database to provide data visualisations & uses the model to classify new messages for 36 different categories


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
