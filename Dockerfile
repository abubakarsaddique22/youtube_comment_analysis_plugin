FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

COPY flask_app/ /app/

COPY models/vectorizer.pkl /app/vectorizer.pkl

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["python", "app.py"]

# Dockerfile