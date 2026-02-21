from fastapi import FastAPI
import joblib
from pydantic import BaseModel

from sentence_tranformers_predict import predict_text

from predict_multiple_models import load_predictor

import joblib
from sentence_transformers import SentenceTransformer

import joblib
from sentence_transformers import SentenceTransformer

MODEL_PATH = "thai_mooc_sbert_logreg_grid_3.joblib"

_bundle = joblib.load(MODEL_PATH)
_embedder = SentenceTransformer(_bundle["model_name"])
_le = _bundle["label_encoder"]
_clf = _bundle["classifier"]
_norm = _bundle.get("normalize_embeddings", True)

def predict_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input must be a non-empty string.")

    emb = _embedder.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=_norm
    )
    pred_idx = _clf.predict(emb)[0]
    return _le.inverse_transform([pred_idx])[0]


class Course(BaseModel):
    name: str
    description: str

app = FastAPI()

model = joblib.load("thai_mooc_sbert_knn.joblib")
print("✅ Model loaded successfully")

@app.post("/predict")
async def predict_course_category(course: Course):
    predicted_category = predict_text(course.description)

    if predicted_category == 'คอมพิวเตอร์และเทคโนโลยี':
        category_id = "1"
    elif predicted_category == 'ธุรกิจและการบริหารจัดการ':
        category_id = "2"
    elif predicted_category == 'สุขภาพและการแพทย์':
        category_id = "3"
    elif predicted_category == 'ภาษาและการสื่อสาร':
        category_id = "4"

    else:
        category_id = "4"  # หมวดหมู่ไม่รู้จัก

    return {"category_name": predicted_category, "category_id": category_id}


