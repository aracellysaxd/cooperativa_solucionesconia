"""Memoria del agente.

  - Corto plazo (STM): el historial de la conversación vive en el estado de LangGraph y se persiste por
    `thread_id` con un checkpointer (ver grafo.py). Cuando crece, se compacta en un resumen.
  - Largo plazo (LTM): episodios por unidad guardados en una colección Chroma y recuperados por
    similitud semántica, filtrando por unidad (así no se mezclan datos entre vecinos).
"""
from datetime import datetime

from langchain_community.vectorstores import Chroma


class MemoriaLargoPlazo:
    def __init__(self, embedding_function, ruta: str = "./memoria_db"):
        self.db = Chroma(
            collection_name="memoria_vecinos",
            persist_directory=ruta,
            embedding_function=embedding_function,
        )

    def guardar(self, unidad: str, texto: str) -> None:
        self.db.add_texts(
            [texto],
            metadatas=[{"unidad": unidad, "fecha": datetime.now().strftime("%Y-%m-%d %H:%M")}],
        )

    def recuperar(self, unidad: str, consulta: str, k: int = 3) -> str:
        try:
            docs = self.db.similarity_search(consulta, k=k, filter={"unidad": unidad})
        except Exception:  # colección vacía u otro error: el agente sigue sin recuerdos
            return ""
        return "\n".join(f"- ({d.metadata.get('fecha', '')}) {d.page_content}" for d in docs)
