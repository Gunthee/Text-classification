import joblib
from encoder import SentenceTransformerEncoder  # ต้อง import เพื่อให้ pickle หา class เจอ


MODEL_PATH = "model/thai_mooc_st_pipeline2.joblib"


def predict(text):
    model = joblib.load(MODEL_PATH)
    return model.predict([text])[0]
   

if __name__ == "__main__":
    text = "วิชานี้จะสอนเกี่ยวกับการวิเคราะห์ข้อมูลเบื้องต้นและการใช้ Python"

    result = predict(text)

    print("Input:", text)
    print("Predicted label:", result)