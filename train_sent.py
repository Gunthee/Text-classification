import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report

from encoder import SentenceTransformerEncoder


def train_and_save(
    csv_path="training_dataset_cleaned_balanced.csv",
    model_out="model/thai_mooc_st_pipeline2.joblib",
    test_size=0.2,
    random_state=42
):
    # ---------------------------
    # Load dataset
    # ---------------------------
    print("📂 Loading dataset...")
    df = pd.read_csv(csv_path).dropna()

    X = df["description"].astype(str)
    y = df["category_name"]

    # ---------------------------
    # Train/Test split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # ---------------------------
    # Pipeline
    # ---------------------------
    pipeline = Pipeline([
        ("embedder", SentenceTransformerEncoder()),
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
        "clf__C": [0.1, 1.0, 3.0, 10.0],
        "clf__solver": ["lbfgs", "liblinear"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,  # สำคัญ: embedding ไม่ควรใช้ -1 พร้อม grid
        verbose=1
    )

    # ---------------------------
    # Train
    # ---------------------------
    print("🚀 Training pipeline...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("\n✅ Best params:", grid.best_params_)
    print("✅ Best CV f1_macro:", grid.best_score_)

    # ---------------------------
    # Evaluate
    # ---------------------------
    print("\n📊 Evaluating...")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # ---------------------------
    # Save pipeline
    # ---------------------------
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(best_model, model_out)

    print(f"\n💾 Saved model → {model_out}")


if __name__ == "__main__":
    train_and_save()