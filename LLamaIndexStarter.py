import logging
import sys

## setup your OpenAI Key
import os
os.environ["OPENAI_API_KEY"] = "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"
# enable logs to see what happen underneath
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import GPTVectorStoreIndex
from llama_index import download_loader

# we will use this UnstructuredReader to read PDF file
UnstructuredReader = download_loader('UnstructuredReader', refresh_cache=True)
loader = UnstructuredReader()
# load the data
data = loader.load_data(f'data/FY22_Q1_Consolidated_Financial_Statements.pdf', split_documents=False)

index = GPTVectorStoreIndex.from_documents(data)
query_engine = index.as_query_engine()
response = query_engine.query("What is the operating income?")
print(response)

