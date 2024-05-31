import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = st.text_input("Enter your OpenAI API key", type="password")

# Function to load and process the PDF
def load_and_process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    return docsearch

# Function to answer questions
def answer_question(docsearch, query):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
    response = qa.run(query)
    return response

# Streamlit app
def main():
    st.title("Ask Saoirse the Persian Cat.")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        docsearch = load_and_process_pdf("temp.pdf")

        query = st.text_input("Ask a question about the PDF")

        if query:
            response = answer_question(docsearch, query)
            st.write(response)

if __name__ == "__main__":
    main()
