import numpy as np
import pandas as pd 
import re 

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline

from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.util import normalize, remove_dangling, remove_dup_spaces


class Classifier:
    def __init__(self):
        self.data = None
        self.model = None

    def read_csv(self):

        data = pd.read_csv('thaimooc_courses.csv')
        data = data.dropna()
        self.data = data
        print("Dataset loaded successfully✅.")
        return 

    def _preprocess_data(self,text):

        cleaned_text = text.strip()\
            .replace('\n', ' ')\
            .replace('\r', ' ')\
            .replace('\t', ' ')\
            .replace('  ', ' ')\
            .replace(':','')\
            .replace('.','')\
            .replace('?','')
        cleaned_text = remove_dup_spaces(cleaned_text)
        cleaned_text = re.sub(r"\d", '', cleaned_text)

        tokens = word_tokenize(cleaned_text, engine='newmm')
        tokens = [normalize(i) for i in tokens]
        tokens = [i for i in tokens if i != '']
        tokens = [i for i in tokens if i not in thai_stopwords()]

        return tokens
    
    def train(self):
        print("Training the classifier with the dataset...")

        X = self.data['Course Description']
        y = self.data['Category']


        pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
        min_df=2,               # ignore rare terms
        sublinear_tf=True,      # log-scale term frequency
        lowercase=True,
        analyzer=lambda x: self._preprocess_data(x),
        decode_error='replace'
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",  # handle imbalance
            n_jobs=-1
        )
        )
        ])

        pipeline.fit(X,y)

        self.model = pipeline.fit(X,y)
        print("Training completed✅.")
        score = pipeline.score(X,y)
        print(f"Training Score: {score}")

    def predict(self, text):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        return self.model.predict([text])[0]
    
    def train_model(self):
        self.read_csv()
        self.train()
        return 

    
classifier = Classifier()
classifier.train_model()

text = "วิชานี้จะสอนเกี่ยวกับการวิเคราะห์ข้อมูลเบื้องต้นและการใช้เครื่องมือในการวิเคราะห์ข้อมูล เช่น Excel, Power BI และ Python เพื่อช่วยในการตัดสินใจทางธุรกิจ"

predicted_category = classifier.predict(text)
print(f"Predicted category: {predicted_category}")



    