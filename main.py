from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    filename: str

class BatchRequest(BaseModel):
    filenames: list[str]

app = FastAPI()

@app.get("/")
def home():
    return {"status": "online"}

@app.get("/analyze/{filename}")
def analyze_get(filename: str) -> dict:
    return get_predictions(filename)


@app.post("/api/v1/analyze/")
def analyze_post(request: AnalyzeRequest):
    return get_predictions(request.filename)
    

@app.post("/api/v1/analyze/batch")
def analyze_batch_post(request: BatchRequest):
    return get_batch_predictions(request.filenames)

@app.post("/api/v1/analyze/image")
async def analyze_minimap_image(file:  UploadFile):
    image_bytes = await file.read()
    return send_to_deeplearning_model(file, image_bytes)


def send_to_deeplearning_model(file: UploadFile, image_bytes: bytes):
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(image_bytes),
        "predictions": {"java": 0.54, "python": 0.46, "c++": 0.01}
    }

def analyze_batch(filenames: list):
    meu_dict = {}
    for file in filenames:
        meu_dict[file] = analyze(file)
    return meu_dict


def get_predictions(filename: str):
    if filename.endswith(".java"):
        return {"java": 0.9, "python": 0.05, "go": 0.05}
    elif filename.endswith(".py"):
        return {"java": 0.1, "python": 0.8, "go": 0.1}
    elif filename.endswith(".go"):
        return {"java": 0.1, "python": 0.1, "go": 0.8}
    else:
        return {"java": 0.3, "python": 0.3, "go": 0.3}
    
def get_batch_predictions(filenames: list):
    meu_dict = {}
    for file in filenames:
        meu_dict[file] = get_predictions(file)
    return meu_dict

if __name__ == "__main__":
    print("Eu sou o main.py rodando!")