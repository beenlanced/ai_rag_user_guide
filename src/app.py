from build_vector_store import FAISS_INDEX, MODEL

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama.llms import OllamaLLM

from operator import itemgetter
import streamlit as st


# Make sure that the Ollama server is running.

# Need to pull the llam3.2 model if I don't already have it.
# I have the model so the line of code below is commented out.
#ollama.pull(MODEL) 

# Create the vector database
ollama_endpoint = "http://127.0.0.1:11434"
embeddings = OllamaEmbeddings(model=MODEL, base_url=ollama_endpoint)

# Get stored Vector Store
vector_store = FAISS.load_local(
    FAISS_INDEX,
    embeddings,
    allow_dangerous_deserialization=True
)

# Setup a Receiver
retriever = vector_store.as_retriever()

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