import logging
import sys
import os
os.environ["OPENAI_API_KEY"] = "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"

## load index
from llama_index import load_index_from_storage
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.index_store import SimpleIndexStore

## create ChromaClient again
import chromadb
from chromadb.config import Settings
chroma_client = chromadb.Client(
        Settings(chroma_db_impl="duckdb+parquet",
                 persist_directory="./storage/vector_storage/chromadb/"
        ))

# load the collection
collection = chroma_client.get_collection("apple_financial_report")

## construct storage context
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext, StorageContext, download_loader, LLMPredictor
from langchain.chat_models import ChatOpenAI

load_storage_context = StorageContext.from_defaults(
    vector_store=ChromaVectorStore(chroma_collection=collection),
    index_store=SimpleIndexStore.from_persist_dir(persist_dir="./storage/index_storage/apple/"),
)

## init LLM Model
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.2, max_tokens=512, model_name='gpt-3.5-turbo'))

## init embedding model
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

## construct service context
load_service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor,embed_model=embed_model)

## finally to load the index
load_index = load_index_from_storage(service_context=load_service_context, 
                                     storage_context=load_storage_context)

query = load_index.as_query_engine()
print(query.query("What is the operating income of Q2 2023?"))