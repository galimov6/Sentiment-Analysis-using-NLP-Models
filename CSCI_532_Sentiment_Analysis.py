import warnings
warnings.filterwarnings("ignore")


import kaggle
import pandas as pd
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.sql import Row
from sparknlp.pretrained import PretrainedPipeline
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import pdb
import datetime
import os
def download_data(path_download = './data/'):

    PATH_file = f'{path_download}/trip_advisor_restaurents_10k_-_trip_rest_neywork_1.csv'

    if 'trip_advisor_restaurents_10k_-_trip_rest_neywork_1.csv' in os.listdir(path_download):
        df = pd.read_csv(PATH_file)
        print(f'Dimension of the data {df.shape}')
        return df
    
    # Make sure to download the kaggle token from your kaggle profile -> Settings - > create new token
    # then add that to your path : /home/USER_NAME/.kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('rayhan32/trip-advisor-newyork-city-restaurants-dataset-10k/',\
                                       path=path_download, unzip=True)
    
    df = pd.read_csv(PATH_file)
    print(f'Dimension of the data {df.shape}')
    return df

def setup_SPARK_NLP():
    lemmas = requests.get('https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/lemma-corpus-small/lemmas_small.txt').text
    with open('lemmas_small.txt','w') as f:
        f.write(lemmas)
    sent = requests.get('https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/sentiment-corpus/default-sentiment-dict.txt').text
    with open('default-sentiment-dict.txt','w') as f:
        f.write(sent)
    return None

def spark_nlp(df,spark,REVIEW_COLUMN='Reveiw Comment'):
    #pdb.set_trace()
    if type(df)==pd.DataFrame:
        spark_df = spark.createDataFrame(df)
    else:
        spark_df =df
    # Step 1: Transforms raw texts to `document` annotation
    document_assembler = (
        DocumentAssembler()
        .setInputCol(REVIEW_COLUMN)
        .setOutputCol("document")
    )

    # Step 2: Sentence Detection, which takes the document as input and adds a new column sentence containing the detected sentences
    sentence_detector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")

    # Step 3: Tokenization, which tokenizes the sentences in the new column token
    tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")

    # Step 4: Lemmatization, which lemmatizes the tokens using the dictionary in lemmas_small.txt with a specified key-value delimiter
    lemmatizer= Lemmatizer().setInputCols("token").setOutputCol("lemma").\
                setDictionary("lemmas_small.txt", key_delimiter="->", value_delimiter="\t")

    # Step 5: Sentiment Detection, which takes the lemmatized tokens and the original sentences as input, and it computes the sentiment scores in the sentiment_score column
    sentiment_detector= (
        SentimentDetector()
        .setInputCols(["lemma", "sentence"])
        .setOutputCol("sentiment_score")
        .setDictionary("default-sentiment-dict.txt", ",")
    )

    # Step 6: Finisher, which gives the final sentiment labels after taking in the sentiment scores
    finisher= (
        Finisher()
        .setInputCols(["sentiment_score"]).setOutputCols("sentiment")
    )

    # Define the pipeline
    pipeline = Pipeline(
        stages=[
            document_assembler,
            sentence_detector, 
            tokenizer, 
            lemmatizer, 
            sentiment_detector, 
            finisher
        ]
    )
    result = pipeline.fit(spark_df).transform(spark_df)
    result =result.withColumn("SPARK sentiment", F.concat_ws(",", F.col("sentiment")))
    result = result.drop('sentiment')
    return result

def viviken(df,spark,REVIEW_COLUMN='Reveiw Comment'):
    #pdb.set_trace()
    if type(df)==pd.DataFrame:
        spark_df = spark.createDataFrame(df)
    else:
        spark_df =df
    # Step 1: Transforms raw texts to `document` annotation
    document_assembler = (
        DocumentAssembler()
        .setInputCol(REVIEW_COLUMN)
        .setOutputCol("document")
    )

    # Step 2: Sentence Detection, which takes the document as input and adds a new column sentence containing the detected sentences
    sentence_detector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")

    # Step 3: Tokenization, which tokenizes the sentences in the new column token
    tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")

    # Step 4: Normalization, which normalizes the tokens in the new column normalizer
    normalizer = Normalizer().setInputCols(["token"]).setOutputCol("normalizer")

    # Step 5: Viviken, which takes document and normalizer as inputs and adds a new column containing sentiment analysis scores called viviken
    viviken = ViveknSentimentModel.pretrained().setInputCols(["document", "normalizer"]).setOutputCol("viviken") 

    # Step 6: Finisher, which gives the final sentiment labels after taking in the sentiment scores 
    finisher= (
        Finisher()
        .setInputCols(["viviken"]).setOutputCols("sentiment_viviken")
    )

    # Define the pipeline
    pipeline = Pipeline(
        stages=[
            document_assembler,
            sentence_detector, 
            tokenizer, 
            normalizer, 
            viviken, 
            finisher
        ]
    )
    result_viviken = pipeline.fit(spark_df).transform(spark_df)
    result_viviken = result_viviken.withColumn("Viviken", F.concat_ws(",", F.col("sentiment_viviken")))
    result_viviken = result_viviken.drop('sentiment_viviken')
    return result_viviken

def vader_func(row,REVIEW_COLUMN='Reveiw Comment'):
    analyzer = SentimentIntensityAnalyzer()
  
    # Perform sentiment analysis on the specified column in the row
    vs = analyzer.polarity_scores(row[REVIEW_COLUMN])

    # Determine sentiment based on the compound score
    if vs['compound']>0:
        sentiment = 'positive'#print(row['Reveiw Comment'],vs)
    elif vs['compound']<0 :
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    # add new item to row
    temp = row.asDict()
    temp["VADER sentiment"] = sentiment

    # Creates a new row with the VADER sentiment item added
    return Row(**temp)    
    
def vader_nlp(df,spark,REVIEW_COLUMN='Reveiw Comment'):
    # sentiment analysis based on VADER NLP: https://vadersentiment.readthedocs.io/en/latest/index.html
    # create regular Spark session
    if type(df)==pd.DataFrame:
        spark_df = spark.createDataFrame(df)
    else:
        spark_df =df
      
    # Create an RDD from the Spark DataFrame
    rdd = spark.sparkContext.parallelize(spark_df.collect())
  
    # Apply sentiment analysis using the vader_func function to each row in the RDD
    results = rdd.map(vader_func).collect()
  
    # Get the columns from the original df and append VADER sentiment to it
    cols =df.columns
    cols.append("VADER sentiment")
  
    # Create a new Spark DataFrame from the results and column names
    results = spark.createDataFrame(results, cols)
    return results

if __name__=='__main__':
    # Download the data
    df= download_data()
    # setup SPARK NLP libs
    setup_SPARK_NLP()
    pdb.set_trace()
    # Start Spark Session
    spark = sparknlp.start()
    # perform spark nlp
    start_spark = datetime.datetime.now()
    result = spark_nlp(df,spark,REVIEW_COLUMN='Reveiw Comment')
    end_spark = datetime.datetime.now()
    # perform VADER NLP ()
    start_vader = datetime.datetime.now()
    result_with_vader = vader_nlp(result,spark,REVIEW_COLUMN='Reveiw Comment')
    end_vader= datetime.datetime.now()
    # perform viviken
    start_viviken = datetime.datetime.now()
    result_viviken = viviken(df,spark,REVIEW_COLUMN='Reveiw Comment')
    end_viviken = datetime.datetime.now()
    vk = end_viviken - start_viviken
    sn = end_spark-start_spark
    vn = end_vader - start_vader
    print('First 20 Rows:')
    print(result_with_vader.show())
    print('---------------------------------------------------')
    print(f'SPARK NLP total elapsed time (seconds) {sn.total_seconds()}')
    print(f'VADER NLP total elapsed time (seconds) {vn.total_seconds()}')
    print(f'VIVIKEN MODEL total elapsed time (seconds) {vk.total_seconds()}')
