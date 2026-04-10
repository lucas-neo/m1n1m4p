Oi pessoal!

Essa é uma API simples utilizando FastAPI em Python.
Minha escolha pelo FastAPI foi devido à facilidade de configurar assincronismo. Por exemplo, nas rotas que recebem um POST de imagem, ele cria um async/await para esperar a imagem carregar sem travar o servidor enquanto ela é carregada.

Eu sei que o Prof. Munif vai utilizar outro tipo de arquivo que vai ser repassado para o modelo de IA. Como ainda não sei qual exatamente, essa API consegue receber uma imagem, uma lista de arquivos ou um arquivo só.

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
E depois, rodar o servidor:

```bash
uvicorn main:app --reload
```