import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader



def process_pdf_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            pages = loader.load()

            for page in pages:
                return page


if "vector" not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings(model = "tinyllama",
)
    st.session_state.loader=WebBaseLoader("https://www.basicphone.in/")
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.docs.append(process_pdf_files("/docs_dir"))

    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)    
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:])
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("ChatGroq Demo")
llm=ChatGroq(groq_api_key="groq_api_key",
             model_name="mixtral-8x7b-32768")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt=st.text_input("Input you prompt here")

if prompt:
    response=retrieval_chain.invoke({"input":prompt})
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
    
