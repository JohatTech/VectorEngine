import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

class ModelLoader():
    def __init__(self):
        # Determine active provider ('azure', 'gemini', or 'ollama')
        self.provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        print(f"ModelLoader initialized with provider: {self.provider}")

    def llm_retriever_loader(self, model_name=None):
        if self.provider == "azure":
            # Requires: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION
            deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
            return AzureChatOpenAI(azure_deployment=deployment)
            
        elif self.provider == "gemini":
            # Requires: GOOGLE_API_KEY
            model = model_name or os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
            return ChatGoogleGenerativeAI(model=model)
            
        else: # Default: Ollama
            model = model_name or "qwen2.5:14b"
            return ChatOllama(model=model, num_ctx=30000)

    def llm_embedding_loader(self, model_name=None):
        if self.provider == "azure":
            deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
            return AzureOpenAIEmbeddings(azure_deployment=deployment)
            
        elif self.provider == "gemini":
            model = model_name or os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
            return GoogleGenerativeAIEmbeddings(model=model)
            
        else: # Default: HuggingFace (Local)
            model = model_name or "hiiamsid/sentence_similarity_spanish_es"
            # Automatically detect CUDA to ensure it scales across different local setups
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_kwargs = {'device': device}
            encode_kwargs = {'normalize_embeddings': False, 'batch_size': 32}
            
            return HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        
    def llm_generator_loader(self, model_name=None):
        if self.provider == "azure":
            deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
            return AzureChatOpenAI(azure_deployment=deployment, temperature=0.65)
            
        elif self.provider == "gemini":
            model = model_name or os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
            return ChatGoogleGenerativeAI(model=model, temperature=0.65)
            
        else: # Default: Ollama
            model = model_name or "qwen2.5:14b"
            return ChatOllama(
                model=model,
                seed=32,
                mirostat_eta=0.15,
                temperature=0.65,
                num_ctx=40000,
                num_gpu=64,
                top_k=30,
            )