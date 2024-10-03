from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from vector import dataset_to_vector
from langchain_google_genai import GoogleGenerativeAI
import logging
from dotenv import load_dotenv

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializando o FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Altere para os domínios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

def create_dynamic_prompt(context_type, question_type):
    template = f"""
    Contexto: {{context}}

    Pergunta ({question_type}): {{input}}
    
    Baseado no contexto fornecido e no tipo de pergunta, forneça uma resposta clara e concisa.
    """
    return ChatPromptTemplate.from_template(template)

def create_specific_prompt(context_type, question_type):
    template = f"""
    Você está recebendo informações sobre ({context_type}). Utilize essas informações para responder à pergunta a seguir:

    Contexto:
    {{context}}
    
    Pergunta ({question_type}):
    {{input}}
    
    Para responder de forma precisa, considere as reviews e detalhes fornecidos. Inclua recomendações baseadas nas características do produto e nas preferências do usuário.
    """
    return ChatPromptTemplate.from_template(template)

def initialize_retrieval_chain():
    logger.info("Carregando variáveis de ambiente...")
    load_dotenv()

    logger.info("Configurando o prompt...")
    prompt = create_specific_prompt("Geral", "Pergunta sobre os dados do contexto")
    
    llm = GoogleGenerativeAI(model="gemini-pro")
    document_chain = create_stuff_documents_chain(llm, prompt=prompt)

    logger.info("Convertendo dataset para vetores...")
    retriever = dataset_to_vector('ruanchaves/b2w-reviews01', use_saved_embeddings=False)

    if retriever is None:
        logger.error("Retriever não foi criado corretamente.")
        return None

    logger.info("Criando a retrieval chain...")
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    return retriever_chain

retriever_chain = initialize_retrieval_chain()

def ask_question(retriever_chain, question):
    try:
        response = retriever_chain.invoke({"input": question})

        if 'answer' in response:
            return response['answer']
        else:
            logger.warning("Nenhuma resposta encontrada ou 'answer' não presente na resposta.")
            return "Nenhuma resposta encontrada."
    except Exception as e:
        logger.error(f"Erro ao tentar responder à pergunta: {e}", exc_info=True)
        return "Erro ao processar a pergunta."

# Rota principal da API que recebe a pergunta e retorna a resposta
@app.post("/ask/")
def ask(request: QuestionRequest):
    if not retriever_chain:
        return {"error": "O sistema não foi inicializado corretamente."}

    question = request.question
    answer = ask_question(retriever_chain, question)
    
    return {"question": question, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
