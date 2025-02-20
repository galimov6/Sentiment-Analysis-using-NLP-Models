# CS 532 Final Project: TripAdvisor Restaurant Reviews - Sentiment Analysis using NLP Models
### Team members: Albert Galimov, Sarah Boal, Fiona McGowan, Jordan Jauch

This project aims to perform sentiment analysis on restaurant reviews from Trip Advisor using 3 different models, Spark NLP, Vader NLP, and Viviken. The analysis is performed on the "Trip Advisor New York City Restaurants Dataset (10k)" from Kaggle. The performance of the different models was measured by the accuracy of the sentiment analysis and the latency in execution time.

Setup
Clone the repository:

    bash git clone https://github.com/your-username/restaurant-sentiment-analysis.git
    cd restaurant-sentiment-analysis

Install the required packages:

    bash pip install -r requirements.txt

Download Kaggle dataset:

If you don't have the Kaggle CLI installed, download the dataset manually from Kaggle.
If you have the Kaggle CLI, you can use it to download the dataset:

    bash kaggle datasets download -d rayhan32/trip-advisor-newyork-city-restaurants-dataset-10k
    unzip trip-advisor-newyork-city-restaurants-dataset-10k.zip -d data

Run the sentiment_analysis.py script:

    bash python sentiment_analysis.py

Description

The download_data function downloads the dataset from Kaggle and returns a Pandas DataFrame. If the dataset is already downloaded, it loads it directly.

The setup_SPARK_NLP function downloads necessary files for Spark NLP (lemma and sentiment dictionaries) from the internet and saves them locally.

The spark_nlp function performs sentiment analysis using Spark NLP on the specified column of the DataFrame.

The viviken function uses the Viviken sentiment model for sentiment analysis.

The vader_nlp function utilizes the VADER sentiment analysis tool for sentiment analysis.

The main block of code demonstrates how to use these functions, including measuring the elapsed time for each method.
