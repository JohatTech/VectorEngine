from langchain_core.retrievers import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter # Text splitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader # Text loader
from langchain.document_loaders import DirectoryLoader
from langchain_chroma import Chroma
from chromadb import Client

import os
import hashlib

class VectorDatabase():
    def __init__(self, path_folder ,client, model_embedding, chunk_size, overlap_size, collection_name, glob):

        self.folder_path = path_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = overlap_size
        self.collection_name = collection_name
        self.embedding_function = model_embedding
        self.client = client
        self.glob = glob
        director= f"./licitaciones_db/collections_{collection_name}"
        self.vectorstore = Chroma(collection_name=self.collection_name, client=client, embedding_function=self.embedding_function, persist_directory=director)


    def set_split(self,folder_path, chunk_size, overlap_size):
        loader = DirectoryLoader(folder_path, use_multithreading=True, max_concurrency=os.cpu_count(), show_progress=True, loader_cls= PyMuPDFLoader)
        self.documents = loader.load()
        print(f"number of documents loaded: {len(self.documents)}")
        splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = overlap_size)
        chunks = splitter.split_documents(self.documents)
        return chunks
    
    def ingest_documents(self):
        chunks = self.set_split(self.folder_path, self.chunk_size, self.chunk_overlap)
        print("start adding document to the vectorstore ")
        # Generate deterministic ids based on page content to prevent duplication
        ids = [hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest() for chunk in chunks]
        self.vectorstore.add_documents(chunks, ids=ids)
        print(f"Added/Updated {len(chunks)} chunks in vectorstore.")
        
    def get_retriever(self, retriever_llm, search_kwargs):
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )
        lic_multi_retriever = MultiQueryRetriever.from_llm(retriever, retriever_llm)
        return lic_multi_retriever, self.vectorstore
