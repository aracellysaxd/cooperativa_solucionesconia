"""Grafo del agente (LangGraph): planificación -> decisión -> herramientas -> memoria.

Flujo (ver diagrama en README):
  cargar_memoria -> planificar -> [derivar | preguntar | agente <-> herramientas] -> guardar_memoria -> compactar
"""
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from .herramientas import HERRAMIENTAS_CON_UNIDAD, HERRAMIENTAS_ESCRITURA

MENSAJE_DERIVACION = (
    "Lamento informarle que ese detalle específico no figura en los registros ni actas actuales de nuestra "
    "cooperativa. Para entregarle una orientación adecuada, le sugiero acercarse directamente a la mesa de "
    "consultas local de la directiva o comunicarse con la administración."
)
MAX_ITERACIONES = 5   # tope de ciclos agente<->herramientas (evita bucles infinitos)
UMBRAL_COMPACTAR = 10  # mensajes a partir de los cuales se resume el historial
CONSERVAR = 4          # mensajes recientes que se mantienen tras compactar
INTENCIONES_QUE_SE_RECUERDAN = {"consulta_cuenta", "calcular_multa", "registrar_solicitud", "redactar_comunicado"}


class Plan(BaseModel):
    """Plan estructurado que produce el nodo planificador."""
    intencion: Literal[
        "consulta_normativa", "consulta_cuenta", "calcular_multa",
        "registrar_solicitud", "redactar_comunicado", "conversacion", "fuera_de_alcance",
    ] = Field(description="Intención principal del vecino.")
    prioridad: Literal["alta", "media", "baja"] = Field(
        description="alta: seguridad, emergencias o daños en curso; media: trámites; baja: consultas generales."
    )
    pasos: list[str] = Field(description="Pasos ordenados (por prioridad) que ejecutará el agente.")
    faltan_datos: list[str] = Field(default_factory=list, description="Datos imprescindibles que el usuario no entregó.")
    pregunta_al_usuario: str = Field(default="", description="Pregunta para obtener los datos faltantes.")


class Estado(TypedDict, total=False):
    messages: Annotated[list, add_messages]  # memoria de corto plazo
    unidad: str
    resumen: str        # historial compactado
    memoria_ltm: str    # recuerdos recuperados de largo plazo
    plan: dict
    iteraciones: int
    traza: list         # decisiones del agente en este turno (para mostrar en la UI)


def texto(contenido) -> str:
    """Gemini a veces devuelve el contenido como lista de partes."""
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        return "".join(p if isinstance(p, str) else p.get("text", "") for p in contenido)
    return str(contenido)


PROMPT_PLANIFICADOR = """Eres el planificador de un agente de la cooperativa de vivienda Los Cerezos (Cerro Navia).
La unidad del usuario YA es conocida: {unidad}. No la pidas.
Resumen de la conversación previa: {resumen}
Recuerdos de interacciones anteriores de esta unidad: {memoria}

Clasifica la última petición del vecino y planifica los pasos:
- consulta_normativa: dudas sobre reglamento, actas, acuerdos, mantenciones.
- consulta_cuenta: gastos comunes, deuda, morosidad de SU unidad.
- calcular_multa: cuánto sería una multa por una infracción.
- registrar_solicitud: quiere dejar un reclamo o solicitud (si falta qué ocurrió, marca faltan_datos).
- redactar_comunicado: quiere una carta/aviso redactado.
- conversacion: saludos, agradecimientos o seguimiento que no requiere herramientas.
- fuera_de_alcance: nada que ver con la cooperativa.
Ordena los pasos por prioridad (lo urgente primero)."""

PROMPT_AGENTE = """Eres el asistente virtual oficial de la cooperativa de vivienda Los Cerezos. Tono amable, empático y claro.
Unidad del usuario: {unidad}. Solo puedes consultar y modificar información de ESTA unidad.
Plan (prioridad {prioridad}): {pasos}
Resumen de la conversación previa: {resumen}
Recuerdos de interacciones anteriores de esta unidad:
{memoria}

Reglas estrictas:
1. Para preguntas normativas usa SIEMPRE buscar_en_actas y responde solo con lo que devuelva; cita el artículo o acta.
2. Si buscar_en_actas no trae la información, NO inventes: responde exactamente: "{derivacion}"
3. Para deuda usa consultar_cuenta; para multas usa calcular_multa; para ver reclamos previos usa listar_solicitudes.
4. Antes de usar una herramienta de ESCRITURA (registrar_solicitud, redactar_comunicado): resume lo que vas a guardar
   y pide confirmación. Llama a la herramienta con confirmado=true solo después de que el usuario confirme.
5. Si una herramienta devuelve ERROR, corrige los argumentos o explica el problema; no repitas la misma llamada fallida.
6. Si hay recuerdos relevantes (por ejemplo una solicitud previa), menciónalos para dar continuidad."""


def construir_grafo(llm, herramientas: list, ltm=None):
    por_nombre = {t.name: t for t in herramientas}
    llm_con_herramientas = llm.bind_tools(herramientas)
    planificador = llm.with_structured_output(Plan)

    def ultimo_mensaje_humano(estado) -> str:
        for m in reversed(estado["messages"]):
            if m.type == "human":
                return texto(m.content)
        return ""

    # ---- Nodos ---------------------------------------------------------------------------------
    def cargar_memoria(estado: Estado) -> dict:
        recuerdos = ltm.recuperar(estado["unidad"], ultimo_mensaje_humano(estado)) if ltm else ""
        n = len(recuerdos.splitlines()) if recuerdos else 0
        return {"memoria_ltm": recuerdos, "iteraciones": 0, "plan": {},
                "traza": [f"Memoria largo plazo: {n} recuerdo(s) recuperado(s) para la unidad {estado['unidad']}"]}

    def planificar(estado: Estado) -> dict:
        sistema = SystemMessage(content=PROMPT_PLANIFICADOR.format(
            unidad=estado["unidad"], resumen=estado.get("resumen") or "(sin resumen)",
            memoria=estado.get("memoria_ltm") or "(sin recuerdos)"))
        try:
            plan = planificador.invoke([sistema] + estado["messages"][-6:])
        except Exception:  # si falla el modelo estructurado, plan por defecto (comportamiento seguro)
            plan = Plan(intencion="consulta_normativa", prioridad="media", pasos=["Buscar en actas y reglamento"])
        traza = estado.get("traza", []) + [
            f"Plan: intención={plan.intencion}, prioridad={plan.prioridad}, pasos={plan.pasos}"]
        return {"plan": plan.model_dump(), "traza": traza}

    def enrutar_plan(estado: Estado) -> Literal["derivar", "preguntar", "agente"]:
        plan = estado["plan"]
        if plan["intencion"] == "fuera_de_alcance":
            return "derivar"
        if plan["faltan_datos"]:
            return "preguntar"
        return "agente"

    def derivar(estado: Estado) -> dict:
        return {"messages": [AIMessage(content=MENSAJE_DERIVACION)],
                "traza": estado.get("traza", []) + ["Decisión: fuera de alcance -> derivación a mesa de consultas"]}

    def preguntar(estado: Estado) -> dict:
        pregunta = estado["plan"].get("pregunta_al_usuario") or "¿Podría darme más detalles para ayudarle?"
        return {"messages": [AIMessage(content=pregunta)],
                "traza": estado.get("traza", []) + [f"Decisión: faltan datos {estado['plan']['faltan_datos']} -> pregunta al usuario"]}

    def agente(estado: Estado) -> dict:
        plan = estado["plan"]
        sistema = SystemMessage(content=PROMPT_AGENTE.format(
            unidad=estado["unidad"], prioridad=plan["prioridad"], pasos="; ".join(plan["pasos"]),
            resumen=estado.get("resumen") or "(sin resumen)",
            memoria=estado.get("memoria_ltm") or "(sin recuerdos)", derivacion=MENSAJE_DERIVACION))
        respuesta = llm_con_herramientas.invoke([sistema] + estado["messages"])
        return {"messages": [respuesta], "iteraciones": estado.get("iteraciones", 0) + 1}

    def decidir_tras_agente(estado: Estado) -> Literal["herramientas", "guardar_memoria"]:
        return "herramientas" if getattr(estado["messages"][-1], "tool_calls", None) else "guardar_memoria"

    def herramientas_node(estado: Estado) -> dict:
        traza = list(estado.get("traza", []))
        salidas = []
        for llamada in estado["messages"][-1].tool_calls:
            nombre, args = llamada["name"], dict(llamada["args"])
            if nombre in HERRAMIENTAS_CON_UNIDAD:
                args["unidad"] = estado["unidad"]  # privilegio mínimo: siempre la unidad del usuario
            try:
                resultado = str(por_nombre[nombre].invoke(args))
            except Exception as e:  # el agente recibe el error y puede corregirse
                resultado = f"ERROR al ejecutar {nombre}: {e}"
            if nombre in HERRAMIENTAS_ESCRITURA and not args.get("confirmado"):
                traza.append(f"Escritura bloqueada ({nombre}): falta confirmación del usuario")
            else:
                traza.append(f"Herramienta {nombre} -> {resultado[:90]}")
            salidas.append(ToolMessage(content=resultado, tool_call_id=llamada["id"], name=nombre))
        return {"messages": salidas, "traza": traza}

    def decidir_tras_herramientas(estado: Estado) -> Literal["agente", "respuesta_final"]:
        return "respuesta_final" if estado["iteraciones"] >= MAX_ITERACIONES else "agente"

    def respuesta_final(estado: Estado) -> dict:
        """Tope de iteraciones alcanzado: se cierra sin herramientas con lo logrado hasta ahora."""
        cierre = SystemMessage(content="Se alcanzó el límite de pasos. Responde al vecino con lo que lograste "
                                       "y explica qué quedó pendiente. No llames herramientas.")
        resp = llm.invoke([cierre] + estado["messages"])
        return {"messages": [resp],
                "traza": estado.get("traza", []) + [f"Decisión: tope de {MAX_ITERACIONES} iteraciones -> cierre controlado"]}

    def guardar_memoria(estado: Estado) -> dict:
        if ltm and estado["plan"].get("intencion") in INTENCIONES_QUE_SE_RECUERDAN:
            ltm.guardar(estado["unidad"],
                        f"Consulta: {ultimo_mensaje_humano(estado)[:200]} | Resultado: {texto(estado['messages'][-1].content)[:300]}")
            return {"traza": estado.get("traza", []) + ["Memoria largo plazo: episodio guardado"]}
        return {}

    def compactar(estado: Estado) -> dict:
        msgs = estado["messages"]
        if len(msgs) <= UMBRAL_COMPACTAR:
            return {}
        corte = len(msgs) - CONSERVAR
        while corte < len(msgs) and msgs[corte].type != "human":  # no dejar herramientas huérfanas
            corte += 1
        if corte >= len(msgs):
            return {}
        viejos = msgs[:corte]
        dialogo = "\n".join(f"{m.type}: {texto(m.content)}" for m in viejos if m.type in ("human", "ai") and texto(m.content))
        resumen = llm.invoke([HumanMessage(content=(
            "Resume en máximo 8 líneas esta conversación conservando datos clave (unidad, solicitudes, montos, decisiones).\n"
            f"Resumen previo: {estado.get('resumen', '')}\n{dialogo}"))])
        return {"resumen": texto(resumen.content), "messages": [RemoveMessage(id=m.id) for m in viejos],
                "traza": estado.get("traza", []) + [f"Memoria corto plazo: {len(viejos)} mensajes compactados en resumen"]}

    # ---- Cableado del grafo -----------------------------------------------------------------------
    g = StateGraph(Estado)
    for nombre, fn in [("cargar_memoria", cargar_memoria), ("planificar", planificar), ("derivar", derivar),
                       ("preguntar", preguntar), ("agente", agente), ("herramientas", herramientas_node),
                       ("respuesta_final", respuesta_final), ("guardar_memoria", guardar_memoria),
                       ("compactar", compactar)]:
        g.add_node(nombre, fn)
    g.add_edge(START, "cargar_memoria")
    g.add_edge("cargar_memoria", "planificar")
    g.add_conditional_edges("planificar", enrutar_plan, {"derivar": "derivar", "preguntar": "preguntar", "agente": "agente"})
    g.add_conditional_edges("agente", decidir_tras_agente, {"herramientas": "herramientas", "guardar_memoria": "guardar_memoria"})
    g.add_conditional_edges("herramientas", decidir_tras_herramientas, {"agente": "agente", "respuesta_final": "respuesta_final"})
    g.add_edge("respuesta_final", "guardar_memoria")
    g.add_edge("guardar_memoria", "compactar")
    g.add_edge("compactar", END)
    g.add_edge("derivar", END)
    g.add_edge("preguntar", END)
    return g.compile(checkpointer=MemorySaver())
