from api import app

import uvicorn
from pydantic import BaseModel

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)

#10.232.197.99:8080/predict