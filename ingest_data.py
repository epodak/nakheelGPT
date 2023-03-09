from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
import pickle
import os
from langchain.vectorstores import Chroma

def embed_doc():
    #check data folder is not empty
    if len(os.listdir("data")) > 0:
        loader = DirectoryLoader('data', glob="**/*.*")
        raw_documents = loader.load()
        print(len(raw_documents))
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 3000,
            chunk_overlap  = 0,
            length_function = len,
        )
        print("111")
        documents = text_splitter.split_documents(raw_documents)


        # Load Data to vectorstore
        embeddings = OpenAIEmbeddings()
        print("222")
        vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
        vectorstore.persist()
        vectorstore = None
        print("333")