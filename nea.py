import os
import textwrap  # For formatting text if needed later
import langchain

# import chromadb
# import transformers
import torch
import requests
import json
from transformers import AutoTokenizer
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM

# from embedfunction import embedding_function
from langchain_huggingface import HuggingFaceEmbeddings

# Retrieve tokens from environment variables

# Here, we use a locally pulled model called "deepseek-r1" through OllamaLLM.
# This model will be used later in our RetrievalQA chain.
model = OllamaLLM(model="mistral")


def docs_preprocessing_helper(file):
    """
    Helper function to load and preprocess a PDF file containing data.

    This function performs two main tasks:
      1. Loads the PDF file using PyPDFLoader from LangChain.
      2. Splits the loaded documents into smaller text chunks using CharacterTextSplitter.

    Args:
        file (str): Path to the PDF file.

    Returns:
        list: A list of document chunks ready for embedding and indexing.
    """
    # Load the PDF file using LangChain's PyPDFLoader.
    loader = PyPDFLoader(file)
    docs = loader.load_and_split()

    # Create a text splitter that divides the documents into chunks up to 5000 characters
    # with an overlap of 200 characters between chunks.
    text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=800)
    docs = text_splitter.split_documents(docs)

    return docs


# Preprocess the PDF file and store the document chunks in 'docs'.
docs = docs_preprocessing_helper("cell.pdf")  # parameter = file

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

embedding_function = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
db = Chroma.from_documents(
    docs, embedding_function, persist_directory="my_chroma_db"
)

template = """You are a teaching chatbot. Use only the source data provided to answer.

If the answer is not in the source data or is incomplete, say:
"I’m sorry, but I couldn’t find the information in the provided data."

{context}

"""

prompt = PromptTemplate(template=template, input_variables=["context"])


formatted_prompt = prompt.format(
    context="You are interacting with college students. They will ask you questions related to the file provided. Please answer their specific questions using the provided file."
)

# Define a refine prompt for iterative refinement if new context is provided.
refine_prompt_template = """You are a teaching chatbot. We have an existing answer:
{existing_answer}

We have the following new context to consider:
{context}

Please refine the original answer if there's new or better information.
If the new context does not change or add anything to the original answer, keep it the same.

If the answer is not in the source data or is incomplete, say:
"I’m sorry, but I couldn’t find the information in the provided data."

Question: {question}

Refined Answer:
"""

refine_prompt = PromptTemplate(
    template=refine_prompt_template,
    input_variables=["existing_answer", "context", "question"],
)


chain_type_kwargs = {
    "question_prompt": prompt,
    "refine_prompt": refine_prompt,
    "document_variable_name": "context",
}

chain = RetrievalQA.from_chain_type(
    llm=model,  # The language model (OllamaLLM with deepseek-r1)
    chain_type="refine",  # "refine" iteratively improves the answer based on additional context.
    retriever=db.as_retriever(
        search_kwargs={"k": 5}
    ),  # Retrieve the top 5 relevant documents.
    chain_type_kwargs=chain_type_kwargs,
)

if __name__ == "__main__":
    query = input("How can I help?")
    response = chain.invoke(query)
    print(response)
