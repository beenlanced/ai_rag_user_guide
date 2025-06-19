from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter


# Setting the Constants
data_directory_path = Path(__file__).resolve().parent.parent /"data"
IPOD_PDF = f"{data_directory_path.joinpath("ipod_shuffle_2015_user_guide.pdf")}"
FAISS_INDEX = f"{data_directory_path.joinpath("faiss_index")}"
MODEL = "llama3.2"

# Loading the PDF data source for the RAG model
loader = PyPDFLoader(IPOD_PDF)
pages = loader.load()

# Need to Split PDF into Chunks for ease of tokenization and Semanitc Search
splitter = RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=100)
chunks = splitter.split_documents(pages)

# Need to pull the llam3.2 model if I don't already have it.
# I have the model so the line of code below is commented out.
#ollama.pull(MODEL) 

# Create the vector database
ollama_endpoint = "http://127.0.0.1:11434"
embeddings = OllamaEmbeddings(model=MODEL, base_url=ollama_endpoint)
vector_store = FAISS.from_documents(chunks, embeddings)

# Now Save FAISS Index for later retrieval 
# Useful so, I don't have to recreate it everytime.
vector_store.save_local(FAISS_INDEX)
