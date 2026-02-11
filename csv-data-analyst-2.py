import streamlit as st
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI as OpenAI
from langchain_classic.agents.agent_types import AgentType
import os

df = pd.DataFrame()

st.title("Chat with Your CSV using RAG Agent")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
llms = OpenAI(temperature=0, model_name='gpt-4.1-nano')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())
    st.write("Data Preview:", df.dtypes)
    st.write("Data Preview:", df.shape)

def create_agent(dataframe):
    agent = create_pandas_dataframe_agent(llms, dataframe, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS,allow_dangerous_code=True)
    return agent

# ... inside the if uploaded_file is not None: block ...
agent = create_agent(df)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the CSV data:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = agent.run(prompt) # The agent processes the query
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})