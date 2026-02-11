import streamlit as st
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
import os

# Set your OpenAI API key as an environment variable
# os.environ["OPENAI_API_KEY"] = "your-api-key"
    
@st.cache_resource
def create_vector_store(dataframe):
    # Convert DataFrame to documents for RAG
    loader = DataFrameLoader(dataframe, page_content_column=dataframe.columns[3]) # Use first column as content for simplicity
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and store in vector db
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


st.title("CSV to SQL/Text Query Bot")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df)
    st.write("Data Preview:", df.dtypes)
    st.write("Data Preview:", df.shape)
    vector_store = create_vector_store(df)
    qa_chain = RetrievalQA.from_chain_type(
        ChatOpenAI(model="gpt-3.5-turbo", temperature=0), 
        chain_type="stuff", 
        retriever=vector_store.as_retriever()
    )


question = st.text_input("Ask a question about the data:")
if question:
        try:
            response = qa_chain.invoke({"query": question})
            st.success(response["result"])
        except Exception as e:
            st.error(f"An error occurred: {e}")
