from fastapi import FastAPI
from pydantic import BaseModel
from train import Classifier

classifier = Classifier()
classifier.train_model()

class Course(BaseModel):
    name: str
    description: str | None = None

app = FastAPI()

@app.post("/predict")
async def predict_course_category(course: Course):
    predicted_category = classifier.predict(course.description)

    if predicted_category == 'คอมพิวเตอร์และเทคโนโลยี':
        category_id = "1"
    elif predicted_category == 'ธุรกิจและการบริหาร':
        category_id = "2"
    elif predicted_category == 'การเงินและการลงทุน':
        category_id = "3"
    elif predicted_category == 'การตลาดและการขาย':
        category_id = "4"
    else:
        category_id = "1"  # หมวดหมู่ไม่รู้จัก

    return {"category_name": predicted_category, "category_id": category_id}


