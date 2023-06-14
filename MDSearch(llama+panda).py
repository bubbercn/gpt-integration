# setup your OpenAI Key
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import Tool
import pandas as pd
from llama_index.indices.struct_store import GPTPandasIndex
import os
os.environ["OPENAI_API_KEY"] = "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"


df = pd.read_excel("data/DSSMDOBJINFO.xlsx")

index = GPTPandasIndex(df=df)

# do imports

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
memory = ConversationBufferMemory(memory_key="chat_history")

from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig
query_engine = index.as_query_engine()

while True:
    text_input = input("User: ")
    try:
        response= query_engine.query(text_input)
    except Exception as e:
        response = str(e)
        # if not response.startswith("Could not parse LLM output: `"):
        # raise e
    print(f'Agent: {response}')
