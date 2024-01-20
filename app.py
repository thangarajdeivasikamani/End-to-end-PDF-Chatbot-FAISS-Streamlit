import streamlit as st
from PyPDF2 import PdfReader
from src.helper import download_hugging_face_embeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from src.prompt import *

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text
 

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs = {'device':'cpu'})
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding_function)
    vector_store.save_local("faiss_index")

def get_conversational_chain():  
    model = CTransformers(model="C:\LLM\LLMA2_Chat\End-to-end-PDF-Chatbot-using-Llama2-main\model\llama-2-7b-chat.ggmlv3.q8_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.5})
    
    PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=PROMPT)
    return chain

def user_input(user_question):

    embedding_function = download_hugging_face_embeddings()   
    new_db = FAISS.load_local("faiss_index", embedding_function) 
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()   
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])



def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using LLaM2 ")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()
