from langchain_core.prompts import ChatPromptTemplate
from modules.Models import ModelLoader
from langchain_core.output_parsers import StrOutputParser

deepseek = "deepseek-r1:14b"
quewn = "qwen2.5:14b"
qwq = "qwq:latest"
llama = "llama3.2:3b"
gemma3 = "gemma3:12b"
mistral = "mistral:latest"
model_loader = ModelLoader()
llm = model_loader.llm_generator_loader(mistral)
template= """
Dada a la siguente información de contexto que son los documentos que describe un proyecto de licitacion, es decir el equivalente a un "request for proposal (RFP) o procurement process": {docs}
sin conocimientos previos, responda las siguiente preguntas y da la informacion solicitda en la pregunta, se conciso y detallado en tus respuestas, responde solo con 20 palabras.
query: {Pregunta}
"""
refine_template = """
Continúa respondiendo a las preguntas, da la informacion solicitda en la pregunta, se conciso y detallado en tus respuestas, responde solo con 20 palabras
    "Nueva pregunta:¿{Pregunta}?
    "usando la informacion de contexto de los siguientes Documentos:{docs}

"""

summarize_prompt = ChatPromptTemplate.from_messages(
    [

        ("human",template)
    ]
)
refine_prompt = ChatPromptTemplate.from_messages(
    [
      
        ("human",refine_template)
    ]
)


initial_chain = summarize_prompt | llm | StrOutputParser()
refine_chain = refine_prompt | llm | StrOutputParser()


from abc import ABC, abstractmethod
import re
class ChainTemplate:
    def __init__(self, prompt, llm, retriever):
        self.prompt = prompt
        self.llm = llm
        self.retriever = retriever
        self.chain = self.prompt | self.llm | StrOutputParser()
    @abstractmethod    
    def get_docs(self, content):
        documents = self.retriever.invoke(content)
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        return doc_texts
    @abstractmethod
    def rip_think(self, answer):
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        return answer
    @abstractmethod
    async def generate(self, content):
        respuesta = await self.chain.ainvoke({"input": content})

        return {"respuesta": respuesta, "index":1}

class ChainIniteResume(ChainTemplate):

    async def generate(self, content, config):
        doc_texts = self.get_docs(content)
        respuesta = await self.chain.ainvoke(
            input= {"Pregunta":content, "docs": doc_texts} ,
            config= config,
        )
        return respuesta