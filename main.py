from api import app
from train import Classifier

import uvicorn
from pydantic import BaseModel

classifier = Classifier()

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8080, reload=True)