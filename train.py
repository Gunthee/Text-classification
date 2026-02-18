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

# ---------------------------
# 1️⃣ Preprocessing function
# ---------------------------

from preprocessing import preprocess


# ---------------------------
# 2️⃣ Load Dataset
# ---------------------------

df = pd.read_csv("thaimooc_courses.csv").dropna()

X = df["Course Description"]
y = df["Category"]


# ---------------------------
# 3️⃣ Encode Labels
# ---------------------------

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# ---------------------------
# 4️⃣ Train/Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# ---------------------------
# 5️⃣ Vectorize Text
# ---------------------------

vectorizer = TfidfVectorizer(
    analyzer=preprocess,
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ---------------------------
# 6️⃣ Train Model
# ---------------------------

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train_tfidf, y_train)


# ---------------------------
# 7️⃣ Evaluate
# ---------------------------

y_pred = model.predict(X_test_tfidf)

print("Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))


# ---------------------------
# 8️⃣ Save Everything
# ---------------------------

model_bundle = {
    "vectorizer": vectorizer,
    "model": model,
    "label_encoder": label_encoder
}

joblib.dump(model_bundle, "thai_mooc_model.joblib")

print("Model saved successfully ✅")
