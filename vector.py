from datasets import load_dataset
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
import json
# import os
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

# Carregar stopwords e lemmatizador do spaCy
nlp = spacy.load("pt_core_news_sm")

nltk_stopwords = set(stopwords.words('portuguese'))

# Palavras a serem removidas das stopwords
palavras_para_manter = {
    'não', 'nunca', 'nenhum', 'eu', 'você', 'nosso', 'nossa', 
    'meu', 'minha', 'este', 'esta', 'esse', 'essa', 'aquele', 
    'aquilo', 'isso', 'mas', 'e', 'ou'
}

# Atualiza a lista de stopwords
nltk_stopwords = nltk_stopwords - palavras_para_manter

punctuation_table = str.maketrans('', '', string.punctuation)

def preprocess_text(text):
    doc = nlp(text.translate(punctuation_table))
    cleaned_text = ' '.join([token.lemma_ for token in doc if token.text.lower() not in nltk_stopwords and not token.is_punct])
    return cleaned_text

def dataset_to_vector(dataset_name, use_saved_embeddings=False):
    load_dotenv()

    if use_saved_embeddings:
        
        data = pd.read_json("dados_dataset.json")

        data = data.fillna("Informação não disponível")
        
        data = data.drop_duplicates(subset=['reviewer_id'])
        
        def format_review(row):
            return (
                f"Data de Submissão: {row.get('submission_date', 'Não disponível')}\n"
                f"ID do Revisor: {row.get('reviewer_id', 'Não disponível')}\n"
                f"ID do Produto: {row.get('product_id', 'Não disponível')}\n"
                f"Nome do Produto: {row.get('product_name', 'Não disponível')}\n"
                f"Marca do Produto: {row.get('product_brand', 'Não disponível')}\n"
                f"Categoria do Site LV1: {row.get('site_category_lv1', 'Não disponível')}\n"
                f"Categoria do Site LV2: {row.get('site_category_lv2', 'Não disponível')}\n"
                f"Título da Revisão: {row.get('review_title', 'Não disponível')}\n"
                f"Avaliação Geral: {row.get('overall_rating', 'Não disponível')}\n"
                f"Recomendaria a um Amigo: {row.get('recommend_to_a_friend', 'Não disponível')}\n"
                f"Texto da Revisão: {row.get('review_text', 'Não disponível')}\n"
                f"Ano de Nascimento do Revisor: {row.get('reviewer_birth_year', 'Não disponível')}\n"
                f"Gênero do Revisor: {row.get('reviewer_gender', 'Não disponível')}\n"
                f"Estado do Revisor: {row.get('reviewer_state', 'Não disponível')}\n"
            )

        documents = [Document(page_content=format_review(row)) for _, row in data.head(100).iterrows()]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text_chunks = text_splitter.split_documents(documents)

        unique_chunks = []
        seen_texts = set()
        for chunk in text_chunks:
            if chunk.page_content not in seen_texts:
                seen_texts.add(chunk.page_content)
                unique_chunks.append(chunk)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(unique_chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 100})
        print("Retriever FAISS criado com sucesso.")    
        return retriever
    else:
        print(f"Carregando o dataset: {dataset_name}")
        dataset = pd.read_json('dados_dataset.json')
        dataset = dataset.fillna("Informação não disponível")
        
        dataset = dataset.drop_duplicates(subset=['reviewer_id'])
        dataset = dataset.head(20000)  #dataset = dataset.select(range(min(len(dataset), 1000)))  # Limitar a 4000 itens

        print(f"Total de itens no dataset: {len(dataset)}")

        texts = []
        for _, item in dataset.iterrows():  # Use iterrows() para iterar sobre o DataFrame
            combined_text = " | ".join([ 
                f"produto: {item['product_name']} | "
                f"nome_produto: {item['product_name']} | "
                f"avaliacao: {item['overall_rating']} | "
                f"marca_produto: {item['product_brand']} | "
                f"categoria_site_lv1: {item['site_category_lv1']} | "
                f"categoria_site_lv2: {item['site_category_lv2']} | "
                f"titulo_avaliacao: {item['review_title']} | "
                f"recomendar_ao_amigo: {item['recommend_to_a_friend']} | "
                f"ano_nascimento_revisor: {item['reviewer_birth_year']} | "
                f"genero_revisor: {item['reviewer_gender']} | "
                f"estado_revisor: {item['reviewer_state']}"
            ])

            # Aplicar a limpeza do texto
            cleaned_text = preprocess_text(combined_text)
            texts.append(cleaned_text)

        # Implementar Sliding Window Chunking
        sliding_window_size = 500
        sliding_overlap = 50
        split_documents = []
       
        for text in texts:
            start = 0
            while start < len(text):
                chunk = text[start:start + sliding_window_size]
                split_documents.append(chunk)
                start += (sliding_window_size - sliding_overlap)

        print(f"Total de documentos após split: {len(split_documents)}")

        print("Obtendo embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        document_embeddings = embeddings.embed_documents(split_documents)

        print(f"Embeddings gerados: {len(document_embeddings)}")

        with open('faiss_embeddings.json', 'w') as f:
            json.dump(document_embeddings, f)
            print("Embeddings salvos localmente em faiss_embeddings.json")

        print("Criando a base de vetores com FAISS...")
        try:
            vector = FAISS.from_texts(split_documents, embedding=embeddings)
            retriever = vector.as_retriever(search_kwargs={"k": 40})
            print("Retriever FAISS criado com sucesso.")
        except Exception as e:
            print(f"Erro ao criar a base de vetores com FAISS: {e}")
            return None

        return retriever
