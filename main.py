from fastapi import FastAPI, HTTPException, Header, Depends
from contextlib import asynccontextmanager
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv
import onnxruntime as ort
import numpy as np
import json
import io
import base64
import os
from mock_data import mock_predictions

load_dotenv()

API_KEY = os.environ.get("API_KEY")

def verify_api_key(x_api_key: str = Header()):
    """
    Pega o header X-Api-Key do request e compara com o token.
    Header() diz pro FastAPI: "pega esse valor do header HTTP".
    O nome do parâmetro (x_api_key) vira o header X-Api-Key automaticamente
    (FastAPI converte underscores em hífens).
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Token inválido")

class ImageRequest(BaseModel):
    project: str
    hash: str
    image: str

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


def run_model(target: str, input_array: np.ndarray, top_n: int = 3) -> dict:
    session = ml_models[target]
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_array})

    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()

    sorted_indices = np.argsort(probabilities)[::-1][:top_n]

    predictions = []
    for idx in sorted_indices:
        predictions.append({
            "class": index_to_label[idx],
            "confidence": round(float(probabilities[idx]), 4)
        })

    return {
        "target": target,
        "predictions": predictions
    }


@app.get("/")
def home():
    return {"status": "online"}


@app.post("/api/v1/analyze/image")
def analyze_minimap_image(request: ImageRequest, _=Depends(verify_api_key)):
    try:
        image_bytes = base64.b64decode(request.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Base64 inválido")

    img = Image.open(io.BytesIO(image_bytes))

    if img.size != (128, 128):
        raise HTTPException(status_code=400, detail="Imagem deve ser 128x128")

    input_array = preprocess_image(img)

    type_result = run_model("type", input_array, top_n=3)
    project_result = mock_predictions("project")
    author_result = mock_predictions("author")
    quality_result = mock_predictions("quality")

    return {
        "hash": request.hash,
        "predict": [
            type_result,
            project_result,
            author_result,
            quality_result
        ]
    }
