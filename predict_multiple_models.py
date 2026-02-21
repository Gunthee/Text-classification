import joblib
from sentence_transformers import SentenceTransformer

def load_predictor(model_path: str):
    bundle = joblib.load(model_path)

    embedder = SentenceTransformer(bundle["sbert_model"])
    clf = bundle["classifier"]
    le = bundle["label_encoder"]
    norm = bundle.get("normalize_embeddings", True)

    def predict_text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input must be a non-empty string.")
        emb = embedder.encode([text], convert_to_numpy=True, normalize_embeddings=norm)
        pred_idx = clf.predict(emb)[0]
        return le.inverse_transform([pred_idx])[0]

    return predict_text


if __name__ == "__main__":
    predict = load_predictor("thai_mooc_sbert_naive_bayes.joblib")

    text = "วิชานี้จะสอนเกี่ยวกับการวิเคราะห์ข้อมูลเบื้องต้นและการใช้ Excel และ Python"
    print("Prediction:", predict(text))