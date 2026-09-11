import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# 1. Configuración inicial de la página web
st.set_page_config(page_title="Asistente Los Cerezos", page_icon="🏘️", layout="centered")

st.title("🏘️ Comunidad Solidaria Los Cerezos")
st.write("Asistente virtual oficial para resolver consultas sobre reglamentos, actas y convivencia de la cooperativa.")

# 2. Cargar entorno y base de datos (con caché para que no recargue en cada mensaje)
@st.cache_resource
def inicializar_sistema_rag():
    load_dotenv()
    mi_clave_api = os.getenv("GOOGLE_API_KEY")
    
    # Cargar embeddings y base vectorial persistente
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # Instanciar el modelo Gemini
    llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", 
    temperature=0, 
    api_key=mi_clave_api
    )
    
    # Prompt empático y profesional
    system_prompt = (
    "Eres el asistente virtual oficial de la cooperativa de vivienda Los Cerezos. "
    "Tu rol es atender a los vecinos con un tono amable, empático, claro y servicial.\n\n"
    "Reglas estrictas de respuesta:\n"
    "1. Si la respuesta está en el contexto, explícala de forma natural, estructurada y conversacional.\n"
    "2. Si la información solicitada NO se encuentra en los documentos oficiales, "
    "NUNCA inventes datos, pero **evita sonar robótico**. En su lugar, responde con esta estructura cálida:\n"
    "'Lamento informarle que ese detalle específico no figura en los registros ni actas actuales de nuestra cooperativa. "
    "Para entregarle una orientación adecuada, le sugiero acercarse directamente a la mesa de consultas local de la directiva o comunicarse con la administración.'\n\n"
    "Contexto oficial disponible:\n{context}"
)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

rag_chain = inicializar_sistema_rag()

# 3. Mantener el historial de la conversación en pantalla
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Caja de texto para que el usuario escriba su consulta
if query := st.chat_input("¿Qué le gustaría consultar sobre la cooperativa?"):
    # Guardar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generar respuesta del Agente RAG
    with st.chat_message("assistant"):
        with st.spinner("Buscando en actas y reglamentos oficiales..."):
            respuesta = rag_chain.invoke({"input": query})
            respuesta_texto = respuesta['answer']
            st.markdown(respuesta_texto)
    
    # Guardar respuesta del asistente
    st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})