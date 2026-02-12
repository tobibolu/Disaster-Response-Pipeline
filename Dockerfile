# ---- Build stage: process data and train model ----
FROM python:3.11-slim AS build

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/disaster_messages.csv data/disaster_categories.csv data/
COPY data/process_data.py data/
COPY models/train_classifier.py models/
COPY utils.py .

# Run ETL pipeline
RUN python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

# Train the classifier
RUN python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

# ---- Runtime stage: serve the Flask web app ----
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy trained artifacts from build stage
COPY --from=build /app/data/DisasterResponse.db data/
COPY --from=build /app/models/classifier.pkl models/

# Copy application code
COPY app/ app/
COPY utils.py .

EXPOSE 3001

CMD ["python", "app/run.py"]
