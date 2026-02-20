from fastapi import FastAPI
import joblib
from pydantic import BaseModel

class Course(BaseModel):
    name: str
    description: str | None = None

app = FastAPI()

model = joblib.load("model/thai_mooc_logreg_grid_2.joblib")
print("✅ Model loaded successfully")

@app.post("/predict")
async def predict_course_category(course: Course):
    predicted_category = model.predict([course.description])[0]

    if predicted_category == 'คอมพิวเตอร์และเทคโนโลยี':
        category_id = "1"
    elif predicted_category == 'ธุรกิจและการบริหารจัดการ':
        category_id = "2"
    elif predicted_category == 'สุขภาพและการแพทย์':
        category_id = "3"
    elif predicted_category == 'ภาษาและการสื่อสาร':
        category_id = "4"

    else:
        category_id = "1"  # หมวดหมู่ไม่รู้จัก

    return {"category_name": predicted_category, "category_id": category_id}


