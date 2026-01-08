import os
import glob
import tiktoken
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

MODEL = "gpt-4.1-nano"
db_name = "vector_db"
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

files = glob.glob("data/*.txt")

def number_of_tokens():

    entire_knowledge_base = ""
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            entire_knowledge_base += f.read()
            entire_knowledge_base += "\n\n"
    
    encoding = tiktoken.encoding_for_model(MODEL)
    tokens = encoding.encode(entire_knowledge_base)
    token_count = len(tokens)
    print(f"Total tokens for {MODEL}: {token_count:,}")

def ingest():
    documents = []

    for file_path in files:
        doc_type = os.path.splitext(os.path.basename(file_path))[0]

        loader = TextLoader(
            file_path,
            encoding="utf-8",
        )

        docs = loader.load()
        for doc in docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

    print(f"Loaded {len(documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n", " ", ""])
    chunks = text_splitter.split_documents(documents)

    print(f"Divided into {len(chunks)} chunks")

    

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")



    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")



if __name__ == "__main__":
    ingest()    