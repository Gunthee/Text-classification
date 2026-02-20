import re
import joblib
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.util import normalize, remove_dup_spaces


# 1) Thai preprocess
from preprocessing import preprocess

df = pd.read_csv("training_dataset 2.csv").dropna()

print(df.columns)

X = df['description']
y = df['category_name']

# 2) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Build pipeline (TFIDF + LinearSVC)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer=preprocess,
        decode_error="replace"
    )),
    ("clf", LinearSVC())
])

# ---------------------------
# 4) GridSearchCV params
# ---------------------------
param_grid = {
    # TF-IDF tuning
    "tfidf__min_df": [1, 2, 3],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__sublinear_tf": [True, False],

    # LinearSVC tuning
    "clf__C": [0.1, 1.0, 3.0, 10.0],
    "clf__class_weight": [None, "balanced"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",   # good for multi-class + imbalance
    cv=cv,
    n_jobs=-1,
    verbose=1
)


# ---------------------------
# 5) Train + pick best model
# ---------------------------
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("\n✅ Best params:", grid.best_params_)
print("✅ Best CV f1_macro:", grid.best_score_)


# ---------------------------
# 6) Evaluate on test set
# ---------------------------
y_pred = best_model.predict(X_test)
print("\n📌 Test classification report:")
print(classification_report(y_test, y_pred))


# ---------------------------
# 7) Save best model
# ---------------------------
joblib.dump(best_model, "model/thai_mooc_linearsvc_dataset_2.joblib")
print("\nSaved ✅ thai_mooc_linearsvc_dataset_2.joblib")


# ---------------------------
# 8) Example prediction
# ---------------------------
text = "วิชานี้จะสอนเกี่ยวกับการวิเคราะห์ข้อมูลเบื้องต้นและการใช้ Python"
print("\nPrediction:", best_model.predict([text])[0])


