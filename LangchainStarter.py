import logging
import sys
import os

os.environ["OPENAI_API_KEY"] = "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"

## load the PDF using pypdf
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# load the data
loader = PyPDFLoader('data/FY22_Q1_Consolidated_Financial_Statements.pdf')

# the 10k financial report are huge, we will need to split the doc into multiple chunk.
# This text splitter is the recommended one for generic text. It is parameterized by a list of characters. 
# It tries to split on them in order until the chunks are small enough.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
data = loader.load()
texts = text_splitter.split_documents(data)

# import Chroma and OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# initialize OpenAIEmbedding
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# use Chroma to create in-memory embedding database from the doc
docsearch = Chroma.from_documents(texts, embeddings,  metadatas=[{"source": str(i)} for i in range(len(texts))])

## perform search based on the question
query = "What is the operating income?"
docs = docsearch.similarity_search(query)

## importing necessary framework
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain

from langchain.chat_models import ChatOpenAI

## use LLM to get answering

# 1
chain = load_qa_chain(ChatOpenAI(temperature=0.2,model_name='gpt-3.5-turbo'), 
                      chain_type="stuff")
print(chain.run(input_documents=docs, question=query))

# 2
chain = load_qa_with_sources_chain(ChatOpenAI(temperature=0.2,model_name='gpt-3.5-turbo'), 
                                   chain_type="stuff")
print(chain({"input_documents": docs, "question": query}, return_only_outputs=True))

# 3
qa=RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0.2,model_name='gpt-3.5-turbo'), chain_type="stuff", 
                                                retriever=docsearch.as_retriever())
print(qa.run(query))

#4
chain=RetrievalQAWithSourcesChain.from_chain_type(ChatOpenAI(temperature=0.2,model_name='gpt-3.5-turbo'), chain_type="stuff", 
                                                    retriever=docsearch.as_retriever())
print(chain({"question": query}, return_only_outputs=True))
