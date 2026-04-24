
from modules.Models import ModelLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

mistral = "mistral:latest"
model_loader = ModelLoader()
llm = model_loader.llm_generator_loader(mistral)
message_prompt = ChatPromptTemplate(
    [
        ("system", "Eres una asistente de analisis de licitaciones, llamada Beatriz La Blanca Inteligencia articial de Applus+ Caribe"),
        ("human","Tu tarea es desarrolla un correo breve de no mas de 20 palabras, donde saludes de manera amable, reiterando que este correo es generado por una IA, y profesional y dando confirmacion del siguiente contexto: {contexto}")
    ]
)
message_chain = message_prompt | llm | StrOutputParser()



async def write_message():
    message =  await message_chain.ainvoke({"contexto":"entrega del analisis de licitiacion que generastes"})
    return message