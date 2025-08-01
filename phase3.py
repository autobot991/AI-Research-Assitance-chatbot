# Phase 1 libraries
# Phase 1 libraries
import os
import warnings
import logging

import streamlit as st

# Phase 2 libraries
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Phase 3 libraries
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Disable warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title('AI  Research Assitance Chatbot!')

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Setup a session state variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Vectorstore setup
@st.cache_resource
def get_vectorstore(pdf_file):
    if pdf_file is None:
        return None

    # Save uploaded file temporarily
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(pdf_file.read())

    # Load and process PDF
    loaders = [PyPDFLoader("temp_uploaded.pdf")]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)

    return index.vectorstore

# Chat input box
prompt = st.chat_input('Pass your prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    # Store user prompt in state
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Phase 2 - prompt template
    groq_sys_prompt = ChatPromptTemplate.from_template("""You are very smart at everything. 
        You always give the most accurate and precise answers. 
        Answer the following Question: {user_prompt}. 
        Start the answer directly. No small talk please.""")

    model = "llama-3.3-70b-versatile"

    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model
    )

    # Phase 3 - vectorstore-based retrieval
    try:
        vectorstore = get_vectorstore(uploaded_file)
        if vectorstore is None:
            st.warning("Please upload a PDF file first.")
        else:
            chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type='stuff',
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True
            )

            result = chain({"query": prompt})
            response = result["result"]

            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
    except Exception as e:
        st.error(f"Error: {str(e)}")
