import re
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report

from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.util import normalize, remove_dup_spaces


# ---------------------------
# Thai preprocessing (must be importable for joblib load)
# ---------------------------
from preprocessing import preprocess

def train_and_save(
    csv_path="training_dataset 2.csv",
    model_out="model/thai_mooc_logreg_grid_2.joblib",
    test_size=0.2,
    random_state=42
):
    # ---------------------------
    # Load dataset
    # ---------------------------
    df = pd.read_csv(csv_path).dropna()
    X = df["description"]
    y = df["category_name"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # ---------------------------
    # Pipeline
    # ---------------------------
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer=preprocess,
            decode_error="replace"
        )),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    # ---------------------------
    # GridSearch parameters
    # ---------------------------
    param_grid = {
        # TF-IDF
        "tfidf__min_df": [1, 2, 3],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__sublinear_tf": [True, False],

        # LogisticRegression
        "clf__C": [0.1, 1.0, 3.0, 10.0],
        "clf__solver": ["lbfgs", "liblinear"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    # ---------------------------
    # Train
    # ---------------------------
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print("\n✅ Best params:", grid.best_params_)
    print("✅ Best CV f1_macro:", grid.best_score_)

    # ---------------------------
    # Evaluate on test set
    # ---------------------------
    y_pred = best_model.predict(X_test)
    print("\n📌 Test classification report:")
    print(classification_report(y_test, y_pred))

    # ---------------------------
    # Save best pipeline (same style as linearsvc joblib)
    # ---------------------------
    joblib.dump(best_model, model_out)
    print(f"\nSaved ✅ {model_out}")


if __name__ == "__main__":
    train_and_save()