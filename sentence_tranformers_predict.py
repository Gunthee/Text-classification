import joblib
from sentence_transformers import SentenceTransformer

MODEL_PATH = "thai_mooc_sbert_logreg_grid.joblib"

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


if __name__ == "__main__":
    text = "วิชานี้จะสอนเกี่ยวกับการวิเคราะห์ข้อมูลเบื้องต้นและการใช้เครื่องมือ เช่น Excel, Power BI และ Python"
    print("Predicted category:", predict_text(text))