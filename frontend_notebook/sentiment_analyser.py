# import libraries
# %matplotlib inline

import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from google.cloud import language_v1 as language


# Set display row/column to show all data
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def analyze_text_sentiment(text):
    """
    This is modified from the Google NLP API documentation found here:
    https://cloud.google.com/natural-language/docs/analyzing-sentiment
    It makes a call to the Google NLP API to retrieve sentiment analysis.
    """
    client = language.LanguageServiceClient()
    document = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)

    response = client.analyze_sentiment(document=document)

    # Format the results as a dictionary
    sentiment = response.document_sentiment
    results = dict(
        text=text,
        score=f"{sentiment.score:.1%}",
        magnitude=f"{sentiment.magnitude:.1%}",
    )

    # Print the results for observation
    for k, v in results.items():
        print(f"{k:10}: {v}")

    # Get sentiment for all sentences in the document
    sentence_sentiment = []
    for sentence in response.sentences:
        item = {}
        item["text"] = sentence.text.content
        item["sentiment score"] = sentence.sentiment.score
        item["sentiment magnitude"] = sentence.sentiment.magnitude
        sentence_sentiment.append(item)

    return sentence_sentiment


def sentiment_analyzer_orchestrator():
    import os
    global text
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
    # define endpoint url
    url = "https://dbgee-mar22-11.ew.r.appspot.com/api/text"
    folder_path = "articles"
    from pathlib import Path
    folder_path = "articles"
    for article in Path(folder_path).glob('*'):
        text = article.read_text(encoding='utf-8')
        # print(text)
        sentiments = analyze_text_sentiment(text)

        print('sentiments', sentiments)


if __name__ == "__main__":
    sentiment_analyzer_orchestrator()
