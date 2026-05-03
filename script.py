"""
Script de teste — manda uma imagem pro endpoint e vê a resposta.
Roda com o servidor ligado (uvicorn main:app --reload)
"""

import base64
import json
import urllib.request

# 1. Ler a imagem e converter pra Base64
img_path = "test_image.png"

with open(img_path, "rb") as f:
    image_bytes = f.read()

image_base64 = base64.b64encode(image_bytes).decode("utf-8")

# 2. Montar o JSON que o endpoint espera
payload = {
    "project": "meu-projeto-teste",
    "hash": "abc123def456",
    "image": image_base64
}

# 3. Mandar o POST
url = "http://127.0.0.1:8000/api/v1/analyze/image"
data = json.dumps(payload).encode("utf-8")

req = urllib.request.Request(
    url,
    data=data,
    headers={
        "Content-Type": "application/json",
        "X-Api-Key": "m1n1m4p-pesquisa-2025"
    }
)
response = urllib.request.urlopen(req)

# 4. Ler e printar a resposta
result = json.loads(response.read())
print(json.dumps(result, indent=2))
