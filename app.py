"""Interfaz Streamlit del agente de la cooperativa Los Cerezos."""
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

from agente.grafo import construir_grafo, texto
from agente.herramientas import crear_herramientas
from agente.memoria import MemoriaLargoPlazo
from agente.servicios import _cargar_unidades

st.set_page_config(page_title="Asistente Los Cerezos", page_icon="🏘️", layout="centered")
st.title("🏘️ Comunidad Solidaria Los Cerezos")
st.caption("Agente virtual: consulta actas y reglamento, revisa su cuenta, calcula multas y registra solicitudes.")

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Falta GOOGLE_API_KEY. Crea un archivo .env con: GOOGLE_API_KEY=tu_clave (ver README).")
    st.stop()


@st.cache_resource
def inicializar_agente():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Base vectorial de actas y reglamento (se crea sola la primera vez)
    db_path = "./chroma_db"
    if not os.path.exists(db_path) or not os.listdir(db_path):
        docs = PyPDFDirectoryLoader("data/actas").load()
        fragmentos = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        vectorstore = Chroma.from_documents(fragmentos, embeddings, persist_directory=db_path)
    else:
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
    herramientas = crear_herramientas(retriever)
    ltm = MemoriaLargoPlazo(embeddings)  # memoria de largo plazo por unidad
    return construir_grafo(llm, herramientas, ltm)


grafo = inicializar_agente()

# --- Barra lateral: identificación de la unidad y control de sesión ---
with st.sidebar:
    st.header("Identificación")
    unidad = st.selectbox("Su unidad", sorted(_cargar_unidades().keys()))
    if "sesion" not in st.session_state:
        st.session_state.sesion = uuid.uuid4().hex[:8]
        st.session_state.mensajes = []
    if st.button("Nueva conversación"):
        st.session_state.sesion = uuid.uuid4().hex[:8]
        st.session_state.mensajes = []
        st.rerun()
    st.caption(f"Sesión: {st.session_state.sesion} (memoria de corto plazo). "
               "Sus solicitudes pasadas se recuerdan entre sesiones (largo plazo).")
    ver_traza = st.toggle("Mostrar razonamiento del agente", value=True)

for m in st.session_state.mensajes:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("traza") and ver_traza:
            with st.expander("Razonamiento del agente"):
                st.markdown("\n".join(f"- {t}" for t in m["traza"]))

if consulta := st.chat_input("¿Qué le gustaría consultar o solicitar?"):
    st.session_state.mensajes.append({"role": "user", "content": consulta})
    with st.chat_message("user"):
        st.markdown(consulta)
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                config = {"configurable": {"thread_id": f"{unidad}-{st.session_state.sesion}"}}
                resultado = grafo.invoke({"messages": [HumanMessage(content=consulta)], "unidad": unidad}, config=config)
                respuesta = texto(resultado["messages"][-1].content)
                traza = resultado.get("traza", [])
            except Exception as e:
                respuesta, traza = f"Ocurrió un error técnico: {e}", []
        st.markdown(respuesta)
        if traza and ver_traza:
            with st.expander("Razonamiento del agente"):
                st.markdown("\n".join(f"- {t}" for t in traza))
    st.session_state.mensajes.append({"role": "assistant", "content": respuesta, "traza": traza})
