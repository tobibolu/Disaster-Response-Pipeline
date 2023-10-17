# Disaster Response Pipeline Project

## Project Summary:
This goal of this project is to create a machine learning pipeline that categorises emergency messages based on the needs communicated by the sender.

### Files:
1. process_data.py - Processes message & category data from CSV files and loans the data into a sqlite database

2. train_classifier.py - Reads from the sqlite database to create and save a multi-output supervised learning model

3. run.py - Web app that extracts data from the database to provide data visualisations & uses the model to classify new messages for 36 different categories
