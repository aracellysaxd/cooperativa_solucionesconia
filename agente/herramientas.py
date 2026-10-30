"""Herramientas del agente (LangChain tools). Envuelven la lógica de servicios.py.

Tipos:
  - CONSULTA  (solo lectura): buscar_en_actas, consultar_cuenta, listar_solicitudes
  - RAZONAMIENTO (cálculo con reglas del reglamento): calcular_multa
  - ESCRITURA (efectos persistentes, exigen `confirmado=True`): registrar_solicitud, redactar_comunicado
"""
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from . import servicios as svc

# Herramientas que operan sobre una unidad: el grafo inyecta la unidad del usuario
# (principio de privilegio mínimo: un vecino solo ve y modifica lo suyo).
HERRAMIENTAS_CON_UNIDAD = {
    "consultar_cuenta", "calcular_multa", "registrar_solicitud", "listar_solicitudes", "redactar_comunicado",
}
HERRAMIENTAS_ESCRITURA = {"registrar_solicitud", "redactar_comunicado"}

TipoMulta = Literal["ruido", "mudanza", "mascotas_excrementos", "estacionamiento_visita", "reserva_espacios_comunes"]


def crear_herramientas(retriever) -> list:
    @tool
    def buscar_en_actas(consulta: str) -> str:
        """CONSULTA. Busca en las actas de asamblea y en el Reglamento de Copropiedad de Los Cerezos
        (normas, multas, mantenciones, acuerdos, morosidad). Úsala para cualquier pregunta normativa."""
        docs = retriever.invoke(consulta)
        if not docs:
            return "SIN_RESULTADOS: no se encontró información en las actas ni en el reglamento."
        return "\n\n".join(
            f"[{Path(d.metadata.get('source', 'doc')).name}, pág. {d.metadata.get('page', '?')}] {d.page_content}"
            for d in docs
        )

    @tool
    def consultar_cuenta(unidad: str) -> str:
        """CONSULTA. Estado de gastos comunes de la unidad: cuotas impagas, deuda, días de mora e interés
        referencial. El parámetro `unidad` se completa automáticamente con la unidad del usuario."""
        return svc.consultar_cuenta(unidad)

    @tool
    def listar_solicitudes(unidad: str) -> str:
        """CONSULTA. Lista las solicitudes/reclamos ya registrados por la unidad (con su estado).
        El parámetro `unidad` se completa automáticamente."""
        return svc.listar_solicitudes(unidad)

    @tool
    def calcular_multa(tipo: TipoMulta, unidad: str) -> str:
        """RAZONAMIENTO. Calcula la multa en UF que correspondería según el reglamento y las infracciones
        previas de la unidad en los últimos 12 meses. `unidad` se completa automáticamente."""
        return svc.calcular_multa(tipo, unidad)

    @tool
    def registrar_solicitud(unidad: str, categoria: str, descripcion: str, confirmado: bool) -> str:
        """ESCRITURA. Registra una solicitud o reclamo (categoría: mantencion, convivencia, seguridad, otro).
        Solo usar `confirmado=True` DESPUÉS de que el usuario confirme explícitamente. `unidad` es automática."""
        return svc.registrar_solicitud(unidad, categoria, descripcion, confirmado)

    @tool
    def redactar_comunicado(unidad: str, asunto: str, cuerpo: str, confirmado: bool) -> str:
        """ESCRITURA. Guarda un borrador de comunicado/carta (por ejemplo aviso de cobro o amonestación).
        Solo usar `confirmado=True` DESPUÉS de que el usuario confirme el borrador. `unidad` es automática."""
        return svc.redactar_comunicado(unidad, asunto, cuerpo, confirmado)

    return [buscar_en_actas, consultar_cuenta, listar_solicitudes, calcular_multa, registrar_solicitud, redactar_comunicado]
