import joblib

# Load once (recommended)
bundle = joblib.load("thai_mooc_model.joblib")

vectorizer = bundle["vectorizer"]
model = bundle["model"]
label_encoder = bundle["label_encoder"]


def predict_text(text: str) -> str:
   
    if not text or not isinstance(text, str):
        raise ValueError("Input must be a non-empty string.")

    # Transform text
    X_tfidf = vectorizer.transform([text])

    # Predict encoded label
    pred_encoded = model.predict(X_tfidf)[0]

    # Convert back to original label
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    return pred_label

text = "วิชานี้จะสอนเกี่ยวกับการวิเคราะห์ข้อมูลเบื้องต้นและการใช้ Python"

result = predict_text(text)

print("Predicted Category:", result)