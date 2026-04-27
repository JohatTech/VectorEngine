import logging
import os
from pathlib import Path
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_classic.agents import create_react_agent, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.tools import create_retriever_tool
from qdrant import QdrantLocalModule, sanitise_collection_name
from core.embeddings import get_embeddings
import config

logger = logging.getLogger("agent_system")

def get_llm():

    chat_provider = config.CHAT_PROVIDER
    deployment = config.AZURE_OPENAI_CHAT_DEPLOYMENT
    api_key = config.AZURE_OPENAI_API_KEY_CHAT
    endpoint = config.AZURE_OPENAI_ENDPOINT_CHAT
    
    if chat_provider == "azure_openai" and api_key:
        # Specialized handling for /openai/responses style endpoints (Global/MaaS)
        if endpoint and "/openai/responses" in endpoint:
            logger.info("Initializing Azure LLM via ChatOpenAI (Global/MaaS)  │  deployment=%s", deployment)
            # Strip query params from endpoint for base_url
            base_url = endpoint.split('?')[0]
            # Ensure it ends with / (some versions need it)
            if not base_url.endswith('/'): base_url += '/'
            
            return ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=deployment,
        
                default_headers={"api-key": api_key}
            )
        
        # Standard Azure OpenAI handling
        from urllib.parse import urlparse
        parsed = urlparse(endpoint or config.AZURE_OPENAI_ENDPOINT)
        base_endpoint = f"{parsed.scheme}://{parsed.netloc}"
        
        logger.info("Initializing AzureChatOpenAI  │  deployment=%s", deployment)
        return AzureChatOpenAI(
            azure_endpoint=base_endpoint,
            api_key=api_key,
            azure_deployment=deployment,
            api_version=config.AZURE_OPENAI_API_VERSION,

        )
    elif api_key:
        model = config.OPENAI_CHAT_MODEL
        logger.info("Initializing ChatOpenAI  │  model=%s", model)
        return ChatOpenAI(
            api_key=api_key,
            model=model,
            
        )
    else:
        raise EnvironmentError("No LLM API keys found (OPENAI_API_KEY or AZURE_OPENAI_API_KEY).")

class AutonomousRAGAgent:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.collection_name = sanitise_collection_name(project_name)
        self.embeddings = get_embeddings()
        self.llm = get_llm()
        
        # Initialize vectorstore connection (local Qdrant as per request)
        # We assume the collection already exists because this runs AFTER vectorization.
        self.qdrant_local = QdrantLocalModule()
        self.vectorstore = self.qdrant_local.get_vectorstore(self.collection_name, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 7})
        
        # Create Retriever Tool
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            "project_document_retriever",
            f"Database of documents and resources that explain every detail about the '{project_name}' project. "
            "You should search and extract accurate information directly from these documents to answer the user's questions."
        )
        self.tools = [self.retriever_tool]
        
        # Define Prompt Template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a helpful assistant in analyzing projects tenders.\n\n"
             "You have access to a database of documents and resources that explain every detail about the project. Your should search and extract accurate information directly from these documents to answer the user's questions.\n\n"
             "The users questions contains description of what the data is about,use it as a guidance to respond.\n\n"
             "Guidelines:\n"
             "- Respond every question ONLY with the information available in the database.\n"
             "- maximum answer of 20 words or less per question.\n"
             "- respond only in Spanish."
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create Agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def process_prompts(self, prompts: list[str]) -> str:
        categories = {
            "Generales": [0, 1, 3, 5],
            "Requisitos Económicos": [2, 6, 7, 18, 19, 20, 21],
            "Requisitos Técnicos": [4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        }
        
        full_report_md = ""
        answers = {}
        
        for i, prompt_text in enumerate(prompts):
            logger.info("Agent  │  Processing prompt %d/%d: '%s...'", i+1, len(prompts), prompt_text[:40])
            
            try:
                response = self.agent_executor.invoke({"input": prompt_text})
                answers[i] = response["output"].strip()
                
            except Exception as e:
                logger.error("Agent  │  Error processing prompt '%s': %s", prompt_text, e)
                answers[i] = f"*Error processing this section: {str(e)}*"
                
        for cat_name, indices in categories.items():
            full_report_md += f"## {cat_name}\n\n"
            for idx in indices:
                if idx in answers and answers[idx]:
                    full_report_md += f"{answers[idx]}\n\n"
                    
        return full_report_md
