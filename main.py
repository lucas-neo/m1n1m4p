from fastapi import FastAPI, UploadFile, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from PIL import Image
import onnxruntime as ort
import numpy as np
import json
import io


class AnalyzeRequest(BaseModel):
    filename: str

class BatchRequest(BaseModel):
    filenames: list[str]

with open("model/type/label_mapping.json") as file:
    label_mapping = json.load(file)

index_to_label = {value: key for key, value in label_mapping.items()}

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["type"] = ort.InferenceSession("model/type/best.onnx")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("L")
    arr = np.array(img, dtype=np.float32)
    arr = arr / 255.0
    arr = arr.reshape(1, 1, 128, 128)
    return arr

def send_to_deeplearning_model(input_array: np.ndarray) -> dict:
    session = ml_models["type"]
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_array})
    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()
    predicted_index = int(np.argmax(probabilities))
    predicted_label = index_to_label[predicted_index]
    confidence = float(probabilities[predicted_index])
    return {
        "predicted_type": predicted_label,
        "confidence": round(confidence, 4),
    }

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

def analyze_batch(filenames: list):
    meu_dict = {}
    for file in filenames:
        meu_dict[file] = get_predictions(file)
    return meu_dict


@app.get("/")
def home():
    return {"status": "online"}

# Rotas de teste — predições mockadas por extensão de arquivo
@app.get("/analyze/{filename}")
def analyze_get(filename: str) -> dict:
    return get_predictions(filename)

@app.post("/api/v1/analyze/")
def analyze_post(request: AnalyzeRequest):
    return get_predictions(request.filename)

@app.post("/api/v1/analyze/batch")
def analyze_batch_post(request: BatchRequest):
    return get_batch_predictions(request.filenames)

# Rota principal — recebe minimap e retorna predição real do modelo ONNX
@app.post("/api/v1/analyze/image")
async def analyze_minimap_image(file: UploadFile):
    image_bytes = await file.read()
    if file.content_type not in ["image/png"]:
        raise HTTPException(status_code=400, detail="Somente PNG")
    img = Image.open(io.BytesIO(image_bytes))
    if img.size != (128, 128):
        raise HTTPException(status_code=400, detail="Imagem deve ser 128x128")
    input_array = preprocess_image(img)
    result = send_to_deeplearning_model(input_array)
    return result

if __name__ == "__main__":
    print("Eu sou o main.py rodando!")
