


from modules.Models import ModelLoader

from modules.chains.EmailMessagesChain import write_message

import re
import os
from pathlib import Path

from modules.Notifiers.EmailNotifier import EmailNotifier
from modules.utils import write_report


model_loader = ModelLoader()
mistral = "mistral:latest"



class LicitacionAnalyzer:
    def __init__(self, document_path, graph):
        self.document_path = document_path
        self.graph_app = graph

        self.path_name = Path(self.document_path)
        self.collection_name = self.path_name.name
        print(self.collection_name)

    async def run(self):

        correo = self.collection_name.split("-")[0]
        name = self.collection_name.split("-")[1]
        name += ".docx"
        print(self.collection_name)
        print(f"nombre de archivo: {name}")
        print(correo)
        thread_id = self.collection_name

        print("INICIANDO GENERACION DE DOCUMENTO...")
        final_text = await self.graph_app.run(thread_id=thread_id)
        
        print("ESCRIBIENDO REPORTE...")
        write_report(final_text, name)

        titulo = await self.graph_app.respond(
            "Título: (En este capítulo poner textualmente el nombre oficial de la licitación)",
            thread_id=thread_id
        )
        print(f"titulo del proyecto: {titulo}")
        
        print("generando mensaje de saludo ")
        message =  await write_message()
        print("ENVIANDO ARCHIVO POR CORREO")
        
        notifier = EmailNotifier()
        notifier.send_report(file_path=name, title=titulo, message=message, correo=correo)
