Oi pessoal!

Essa é a API do m1n1m4p, feita com FastAPI em Python.
Ela recebe uma imagem minimap (PNG 128x128 grayscale) via JSON/Base64 e retorna predições usando modelos ONNX.

Hoje o modelo de `type` (linguagem de programação) é real, os de `project`, `author` e `quality` ainda são mock.

Para usar, é só ter Python instalado.
Habilite um ambiente Python antes de instalar as dependências com o comando:

```bash
python3 -m venv venv
source venv/bin/activate
```

Com o ambiente habilitado, é só rodar:

```bash
pip install -r requirements.txt
```

Para subir o servidor:

```bash
uvicorn main:app --reload
```

A API fica em `http://127.0.0.1:8000`. A documentação automática fica em `http://127.0.0.1:8000/docs`.

Para rodar com Docker:

```bash
docker build -t m1n1m4p .
docker run -p 8000:8000 -e API_KEY=seu-token-aqui m1n1m4p
```

Todas as rotas (exceto `/`) precisam do header `X-Api-Key` com o token configurado na variável de ambiente `API_KEY`.

Para testar, com o servidor rodando:

```bash
python3 script.py
```
