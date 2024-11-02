import os
from dotenv import load_dotenv
from initialization import load_data, initialize_text_splitter, initialize_embeddings, initialize_faiss
from langchain_community.llms import HuggingFaceEndpoint

load_dotenv() 
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# Загружаем и подготавливаем данные
data_path = '/tg_bot/formatted_data.json' 
data = load_data(data_path)

# Инициализируем разделитель и создаём документы
text_splitter = initialize_text_splitter()
texts = text_splitter.create_documents(data)

# Инициализируем модель эмбеддинга, базу данных и llm модель
hf_embeddings_model = initialize_embeddings()
retriever = initialize_faiss(texts, hf_embeddings_model)
llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-72B-Instruct")
