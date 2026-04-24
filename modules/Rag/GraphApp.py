from langchain_core.runnables import RunnableConfig
from tqdm import tqdm
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from modules.utils import write_report

from typing import List, Literal, TypedDict
import re

class State(TypedDict):
    contents: List[str]
    index: int 
    respuesta: str


class GraphRAGApp:
    def __init__(self, prompts: List[str], init_chain, refine_chain, lic_multi_retriever):
        self.initial_sumary_chain = init_chain

        self.refine_summary_chain = refine_chain  
        self.lic_multi_retriever =lic_multi_retriever
        self.prompts = prompts  

        self.memory = MemorySaver()
        self.graph = StateGraph(State)
        self.graph.add_node("generate_initial_respond", self.generate_initial_respond)
        self.graph.add_node("refine_respond", self.refine_respond)
        self.graph.add_edge(START, "generate_initial_respond")
        self.graph.add_conditional_edges("generate_initial_respond", self.should_refine)
        self.graph.add_conditional_edges("refine_respond", self.should_refine)
        self.app = self.graph.compile(checkpointer=self.memory)
        
    def get_config(self, thread_id: str):
        return {
            "configurable": {
                "thread_id": thread_id,
            },  
            "recursion_limit": 100
        }

    
    def get_docs(self, content):
        documents = self.lic_multi_retriever.invoke(content)
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        return doc_texts
    
    def rip_think(self, answer):
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        return answer
    
    def show_graph(self):
        from IPython.display import Image
        return Image(self.app.get_graph().draw_mermaid_png())
    
    #FIRST NODE 
    async def generate_initial_respond(self, state:State, config:RunnableConfig):
        doc_texts = self.get_docs(state["contents"][0])
        respuesta = await self.initial_sumary_chain.ainvoke(
            input= {"Pregunta":state["contents"][0], "docs": doc_texts} ,
            config= config,
        )
        
        respuesta = self.rip_think(respuesta)
            
        return {"respuesta": respuesta, "index":1}
    #SECOND NODE 
    async def refine_respond(self, state:State, config:RunnableConfig):
        content = state["contents"][state["index"]]
        doc_texts = self.get_docs(content)

        #generating nexts answers, without giving previous answers as input 
        respuesta = await self.refine_summary_chain.ainvoke(
            {"Pregunta": content, "docs":doc_texts},
            config,
        ) 
        
        respuesta = self.rip_think(respuesta)
        print(f"Question: {respuesta}")
        return {"respuesta": respuesta, "index": state["index"] + 1}

    def should_refine(self, state:State) ->Literal["refine_respond", END]:
        if state["index"] >= len(state["contents"]):
            return END
        else:
            return "refine_respond"
        
    async def run(self, thread_id: str = "default_thread"):
        full_output = []
        config = self.get_config(thread_id)
        async for step in self.app.astream( 
            {"contents": self.prompts},
            stream_mode="values",
            config=config
        ):
            if output := step.get("respuesta"):
                print(output, flush=True)
                full_output.append(output)
        
        final_text = "".join(full_output)
        return final_text

        
    async def respond(self, question, thread_id: str = "default_thread"):
        doc_texts = self.get_docs(question)
        config = self.get_config(thread_id)
        try:
            respuesta = await self.initial_sumary_chain.ainvoke(
                input={"Pregunta": question, "docs": doc_texts},
                config=config
            )
            
            respuesta = self.rip_think(respuesta)
                
            return respuesta
        except BaseException as e:
            print(f"error in respond:{e}")