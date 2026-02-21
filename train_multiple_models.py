import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def train_sbert_models(
    csv_path: str = "thaimooc_courses.csv",
    text_col: str = "Course Description",
    label_col: str = "Category",
    sbert_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    normalize_embeddings: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
):
    # 1) Load dataset
    df = pd.read_csv(csv_path).dropna(subset=[text_col, label_col])
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()

    # 2) Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # 3) Split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 4) Embeddings (compute once, reuse for all models)
    embedder = SentenceTransformer(sbert_model)
    X_train_emb = embedder.encode(
        X_train_text,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    X_test_emb = embedder.encode(
        X_test_text,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # 5) Define models + grids
    models = {
        "svm": (
            LinearSVC(),
            {
                "C": [0.1, 1.0, 3.0, 10.0],
                "class_weight": [None, "balanced"],
            }
        ),
        "random_forest": (
            RandomForestClassifier(random_state=random_state, n_jobs=-1),
            {
                "n_estimators": [200, 500],
                "max_depth": [None, 20, 50],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            }
        ),
        "knn": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 9, 15],
                "weights": ["uniform", "distance"],
                "metric": ["cosine", "euclidean"],
            }
        ),
        # For SBERT embeddings, GaussianNB is the safe NB choice.
        "naive_bayes": (
            GaussianNB(),
            {
                "var_smoothing": [1e-9, 1e-8, 1e-7]
            }
        )
    }

    # 6) Train + evaluate + save each best model
    for name, (estimator, param_grid) in models.items():
        print("\n" + "=" * 70)
        print(f"Training model: {name}")

        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train_emb, y_train)
        best_model = grid.best_estimator_

        print("✅ Best params:", grid.best_params_)
        print("✅ Best CV f1_macro:", grid.best_score_)

        y_pred = best_model.predict(X_test_emb)
        print("\n📌 Test classification report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        out_path = f"thai_mooc_sbert_{name}.joblib"

        # Save a small, robust bundle (don’t pickle the embedder object)
        bundle = {
            "sbert_model": sbert_model,
            "normalize_embeddings": normalize_embeddings,
            "label_encoder": le,
            "classifier_name": name,
            "classifier": best_model,
        }
        joblib.dump(bundle, out_path)
        print(f"Saved ✅ {out_path}")


if __name__ == "__main__":
    train_sbert_models()