# Use a imagem oficial do Python como base
# FROM python:3.9
FROM python:3.12

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia o arquivo de dependências
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da aplicação para o contêiner
COPY . .

# Expõe a porta em que a aplicação será executada
EXPOSE 8000

# Comando para iniciar a aplicação FastAPI
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uvicorn", "main:app", "--reload"]