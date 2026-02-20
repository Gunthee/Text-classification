import joblib

model = joblib.load("model/thai_mooc_logreg_grid_2.joblib")

text = "วิชานี้จะสอนเกี่ยวกับการวิเคราะห์ข้อมูลเบื้องต้นและการใช้ Python"

pred_encoded = model.predict([text])[0]

print("Predicted encoded label:", pred_encoded)