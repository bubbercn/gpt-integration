## setup your OpenAI Key
import os
os.environ["OPENAI_API_KEY"] = "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"

from llama_index.indices.struct_store import GPTPandasIndex
import pandas as pd

df = pd.read_excel("data/DSSMDOBJINFO.xlsx")

index = GPTPandasIndex(df=df)

# do imports
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig
query_engine = index.as_query_engine()
tool_config = IndexToolConfig(
    query_engine=query_engine, 
    name=f"Metadata Info",
    description=f"useful for when you want to answer queries about the Metadata",
    tool_kwargs={"return_direct": True}
)
toolkit = LlamaToolkit(
    index_configs=[tool_config]
)
memory = ConversationBufferMemory(memory_key="chat_history")
llm=ChatOpenAI(temperature=0.2,model_name='gpt-3.5-turbo')
agent_chain = create_llama_chat_agent(
    toolkit,
    llm,
    memory=memory,
    verbose=True
)
while True:
    text_input = input("User: ")
    try:
        response = agent_chain.run(input=text_input)
    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse LLM output: `"):
            raise e
    print(f'Agent: {response}')