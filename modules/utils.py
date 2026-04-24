from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import TextLoader
import re
from docx import Document 
def load_doc_input(path, doc_type, chunk_size, chunk_overlap ):
    if doc_type == "xlsx":
       loader = UnstructuredExcelLoader(path)
    elif doc_type == "docx":
       loader = UnstructuredWordDocumentLoader(path)
    elif doc_type == "pdf":
        loader = PyMuPDFLoader(path,mode = "page", pages_delimiter = " ")
    elif doc_type == "txt":
        loader =TextLoader(path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(separators=["\n"], chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunks = splitter.split_documents(documents)
    return documents, chunks 
import os 
def get_text_file(example_path):
    with open(example_path, encoding="utf8") as file:
        data = file.readlines()
    texts = " ".join(data)
    return texts
def write_report(text, file_name):
    try:
        doc = Document()
        doc.add_paragraph(text)
        print(f"writing file name: {file_name}")
        doc.save(file_name)
    except BaseException as e:
        print(f"error writing report: {e}")
def set_collection_name(name):
    name =name.lower()
    name = re.sub(r"[\s\-]+", "_", name)
    # Elimina cualquier carácter que no sea letra, número o guión bajo
    name = re.sub(r"[^a-z0-9_]", "", name)
    # Quita guiones bajos al principio o final (opcional)
    name = name.strip("_")
    return name
