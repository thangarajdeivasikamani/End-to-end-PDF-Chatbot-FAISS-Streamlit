# from langchain.document_loaders.csv_loader  import UnstructuredCSVLoader
# from langchain_community.document_loaders import UnstructuredExcelLoader
# from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


#Extract data from the CSV
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyMuPDFLoader,
                    )
    
    documents = loader.load()

    return documents


#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks



#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs = {'device':'cpu'},encode_kwargs={'normalize_embeddings':False})
    return embeddings