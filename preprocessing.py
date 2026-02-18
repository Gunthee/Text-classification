import re
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.util import normalize, remove_dup_spaces

stopwords = set(thai_stopwords())

def preprocess(text):
    text = str(text).strip()\
        .replace('\n', ' ')\
        .replace('\r', ' ')\
        .replace('\t', ' ')\
        .replace(':','')\
        .replace('.','')\
        .replace('?','')

    text = remove_dup_spaces(text)
    text = re.sub(r"\d+", "", text)

    tokens = word_tokenize(text, engine='newmm')
    tokens = [normalize(t) for t in tokens if t]
    tokens = [t for t in tokens if t not in stopwords]

    return tokens