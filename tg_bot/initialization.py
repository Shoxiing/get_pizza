from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import json
import torch

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result_str = [f"{item['query']}: {item['answer']}" for item in data[1:]]
    return result_str


def initialize_text_splitter():
    return RecursiveCharacterTextSplitter(
        separators=["{", "}", "\n\n", "\n", " ", ""],
        chunk_size=250,
        chunk_overlap=0
    )

def initialize_embeddings(model_name="sergeyzh/LaBSE-ru-turbo", device="cuda", ):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device}
    )

def initialize_faiss(texts, embeddings_model):
    db = FAISS.from_documents(texts, embeddings_model)
    return db.as_retriever(search_type="similarity", k=10)
