import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

MODEL_OUT = "thai_mooc_sbert_logreg_grid_3.joblib"

def train_sbert_logreg_grid(
    csv_path: str = "training_dataset_cleaned_balanced.csv",
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    test_size: float = 0.2,
    random_state: int = 42,
):
    # 1) Load data
    df = pd.read_csv(csv_path).dropna()
    X_text = df["description"].astype(str).tolist()
    y = df["category_name"].astype(str).tolist()

    # 2) Encode labels (optional but nice for storing + reporting)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 3) Train/test split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y_enc,
        test_size=test_size,
        random_state=random_state,
        stratify=y_enc
    )

    # 4) Sentence embeddings (SBERT)
    embedder = SentenceTransformer(model_name)
    X_train_emb = embedder.encode(
        X_train_text,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # helps for linear models often
    )
    X_test_emb = embedder.encode(
        X_test_text,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # 5) Classifier + GridSearchCV
    base_clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        n_jobs=-1,
        solver="lbfgs"
    )

    param_grid = {
        "C": [0.1, 1.0, 3.0, 10.0],
        # You can also tune class_weight if you want:
        # "class_weight": [None, "balanced"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train_emb, y_train)

    best_clf = grid.best_estimator_
    print("\n✅ Best params:", grid.best_params_)
    print("✅ Best CV f1_macro:", grid.best_score_)

    # 6) Evaluate on test
    y_pred = best_clf.predict(X_test_emb)
    print("\n📌 Test classification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 7) Save bundle (DON'T save embedder object; save model name instead)
    bundle = {
        "model_name": model_name,
        "label_encoder": le,
        "classifier": best_clf,
        "normalize_embeddings": True,
    }
    joblib.dump(bundle, MODEL_OUT)
    print(f"\nSaved ✅ {MODEL_OUT}")


if __name__ == "__main__":
    train_sbert_logreg_grid()