from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama import ChatOllama
from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter

import ollama
from operator import itemgetter
import streamlit as st


# Make sure that the Ollama server is running.

# Setting the Constants
data_directory_path = Path(__file__).resolve().parent.parent /"data"
IPOD_PDF = f"{data_directory_path.joinpath("ipod_shuffle_2015_user_guide.pdf")}"
MODEL = "llama3.2"

# Loading the PDF data source for the RAG model
loader = PyPDFLoader(IPOD_PDF)
pages = loader.load()

# Need to Split PDF into Chunks for ease of tokenization and Semanitc Search
splitter = RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=100)
chunks = splitter.split_documents(pages)

# Need to pull the llam3.1 model
ollama.pull(MODEL)

# Create the vector database
ollama_endpoint = "http://127.0.0.1:11434"
embeddings = OllamaEmbeddings(model=MODEL, base_url=ollama_endpoint)
vectorstore = FAISS.from_documents(chunks, embeddings)

# Setup a Receiver
retriever = vectorstore.as_retriever()

# Creating the model
model = OllamaLLM(model=MODEL)

# Parsing the model's response
parser = StrOutputParser()

# Creating a Template
template = """
You are an assistant that provides answers to questions based on a given context.

Answer the question based on the context.
If you can't anser the question, reply "I do not know".

Be as concise as possible and go straight to the point.

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template=template)

# Adding the Retriever to the Chain
chain = (
  {
    "context": itemgetter("question") | retriever,
    "question": itemgetter("question"),
  }
  | prompt
  | model
  | parser
)

# Setting up the StreamLit App
st.title("The iPod Shuffle 2015 User Guide App")

question = st.chat_input("Enter your user guide question here:")
if question: 
    st.write(chain.invoke({"question": question}))