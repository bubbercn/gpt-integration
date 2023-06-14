import logging
import sys
import os
os.environ["OPENAI_API_KEY"] = "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"

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext, StorageContext, download_loader, LLMPredictor
from langchain.chat_models import ChatOpenAI
from llama_index.vector_stores import ChromaVectorStore

import chromadb
from chromadb.config import Settings

# init Chroma collection
chroma_client = chromadb.Client(
          Settings(chroma_db_impl="duckdb+parquet",
           persist_directory="./storage/vector_storage/chromadb/"
   ))

## create collection
chroma_collection = chroma_client.create_collection("apple_financial_report")

## I use OpenAI ChatAPT as LLM Model. This will cost you money
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.2, max_tokens=512, model_name='gpt-3.5-turbo'))

## by default, LlamIndex use OpenAI's embedding, we will use HuggingFace's embedding instead
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

## init ChromaVector storage for storage context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

## init service context
service_context = ServiceContext.from_defaults(
      llm_predictor=llm_predictor,
      embed_model=embed_model
)

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

# load document
documents = SimpleDirectoryReader(input_files=[f'data/FY23_Q2_Consolidated_Financial_Statements.pdf']).load_data()
# use GPTVectorStoreIndex, it will call embedding mododel and store the 
# vector data (embedding data) in the your storage folder
index = GPTVectorStoreIndex.from_documents(documents=documents, 
                                           storage_context=storage_context,
                                           service_context=service_context)

## save index
index.set_index_id("gptvector_apple_finance")
index.storage_context.persist('./storage/index_storage/apple/')

