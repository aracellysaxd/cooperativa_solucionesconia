"""Lógica de negocio pura (sin LLM): fácil de probar y de explicar en la defensa.

Aquí viven las reglas del Reglamento de Copropiedad (multas, morosidad) y los
efectos de escritura (solicitudes y comunicados). Las herramientas del agente
(herramientas.py) solo envuelven estas funciones.
"""
import csv
import sqlite3
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RUTA_UNIDADES = BASE / "data" / "sistema" / "unidades.csv"
RUTA_DB = BASE / "data" / "sistema" / "solicitudes.db"
DIR_SALIDAS = BASE / "salidas"

TOPE_MULTA_UF = 3.0  # Art. 46 del Reglamento

# Art. 24 (ruido): 1ª amonestación, 2ª 0,3 UF, 3ª 0,6 UF, 4ª y siguientes 1,0 UF
ESCALA_RUIDO = [0.0, 0.3, 0.6, 1.0]
TIPOS_MULTA = {
    "ruido": "Art. 24",
    "mudanza": "Art. 25",
    "mascotas_excrementos": "Art. 27",
    "estacionamiento_visita": "Art. 34",
    "reserva_espacios_comunes": "Art. 33",
}


def _cargar_unidades() -> dict:
    with open(RUTA_UNIDADES, encoding="utf-8") as f:
        return {fila["unidad"]: fila for fila in csv.DictReader(f)}


def _clp(valor: int) -> str:
    """Formatea pesos chilenos con punto de miles: 262000 -> $262.000."""
    return "$" + f"{valor:,}".replace(",", ".")


def consultar_cuenta(unidad: str) -> str:
    """Estado de gastos comunes de una unidad (herramienta de CONSULTA)."""
    fila = _cargar_unidades().get(unidad)
    if fila is None:
        return f"ERROR: la unidad '{unidad}' no existe en el sistema."
    cuotas = int(fila["cuotas_impagas"])
    monto = int(fila["monto_adeudado_clp"])
    dias = int(fila["dias_mora"])
    if cuotas == 0:
        return f"Unidad {unidad}: al día, sin cuotas impagas."
    interes_min = round(monto * 0.010 * dias / 30)
    interes_max = round(monto * 0.015 * dias / 30)
    notas = []
    if dias > 60:
        notas.append("Art. 15: la Administración puede cargar un recargo de cobranza de hasta 0,2 UF (mora > 60 días).")
    if cuotas >= 3:
        notas.append("Art. 18: con 3 o más cuotas impagas el Comité puede suspender servicios comunes, con aviso escrito de 15 días.")
    return (
        f"Unidad {unidad}: {cuotas} cuota(s) impaga(s), capital adeudado {_clp(monto)} CLP, {dias} días de mora. "
        f"Interés moratorio referencial (Art. 14, 1,0%-1,5% mensual simple): {_clp(interes_min)} a {_clp(interes_max)} CLP. "
        + " ".join(notas)
    )


def calcular_multa(tipo: str, unidad: str) -> str:
    """Calcula la multa aplicable según reincidencia en 12 meses (herramienta de RAZONAMIENTO)."""
    if tipo not in TIPOS_MULTA:
        return f"ERROR: tipo '{tipo}' no válido. Opciones: {', '.join(TIPOS_MULTA)}."
    fila = _cargar_unidades().get(unidad)
    if fila is None:
        return f"ERROR: la unidad '{unidad}' no existe en el sistema."
    art = TIPOS_MULTA[tipo]
    if tipo == "ruido":
        previas = int(fila["ruido_12m"])
        multa = ESCALA_RUIDO[min(previas, 3)]
        detalle = f"{previas} infracción(es) previa(s) por ruido en 12 meses -> esta sería la infracción N° {previas + 1}"
        if multa == 0:
            return f"{art}: {detalle}. Corresponde AMONESTACIÓN POR ESCRITO, sin multa pecuniaria."
    elif tipo == "mascotas_excrementos":
        previas = int(fila["mascotas_12m"])
        multa = 0.3 if previas == 0 else 0.6
        detalle = f"{previas} infracción(es) previa(s) por mascotas en 12 meses"
    else:
        fija = {"mudanza": 0.5, "estacionamiento_visita": 0.4, "reserva_espacios_comunes": 0.5}[tipo]
        multa, detalle = fija, "multa fija, sin escala de reincidencia"
    multa = min(multa, TOPE_MULTA_UF)
    return f"{art}: {detalle}. Multa aplicable: {multa} UF (tope por multa individual: {TOPE_MULTA_UF} UF, Art. 46)."


def registrar_solicitud(unidad: str, categoria: str, descripcion: str, confirmado: bool) -> str:
    """Crea un ticket (herramienta de ESCRITURA). Exige confirmación explícita del vecino."""
    if not confirmado:
        return "NO EJECUTADO: falta la confirmación del usuario. Resume la solicitud y pide que confirme."
    if not descripcion.strip():
        return "ERROR: la descripción de la solicitud está vacía."
    RUTA_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(RUTA_DB)
    con.execute(
        "CREATE TABLE IF NOT EXISTS solicitudes (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "unidad TEXT, categoria TEXT, descripcion TEXT, estado TEXT, creada TEXT)"
    )
    cur = con.execute(
        "INSERT INTO solicitudes (unidad, categoria, descripcion, estado, creada) VALUES (?,?,?,?,?)",
        (unidad, categoria, descripcion.strip(), "abierta", datetime.now().isoformat(timespec="seconds")),
    )
    con.commit()
    nuevo_id = cur.lastrowid
    con.close()
    return f"OK: solicitud #{nuevo_id} registrada para la unidad {unidad} (categoría: {categoria}, estado: abierta)."


def listar_solicitudes(unidad: str) -> str:
    """Solicitudes previas de la unidad (herramienta de CONSULTA)."""
    if not RUTA_DB.exists():
        return f"La unidad {unidad} no tiene solicitudes registradas."
    con = sqlite3.connect(RUTA_DB)
    filas = con.execute(
        "SELECT id, categoria, descripcion, estado, creada FROM solicitudes WHERE unidad=? ORDER BY id", (unidad,)
    ).fetchall()
    con.close()
    if not filas:
        return f"La unidad {unidad} no tiene solicitudes registradas."
    return "\n".join(f"#{i} [{est}] {cat}: {desc} ({fecha})" for i, cat, desc, est, fecha in filas)


def redactar_comunicado(unidad: str, asunto: str, cuerpo: str, confirmado: bool) -> str:
    """Guarda un borrador de comunicado en salidas/ (herramienta de ESCRITURA)."""
    if not confirmado:
        return "NO EJECUTADO: falta la confirmación del usuario. Muestra el borrador y pide que confirme."
    DIR_SALIDAS.mkdir(exist_ok=True)
    nombre = f"comunicado_{unidad}_{datetime.now():%Y%m%d_%H%M%S}.md"
    (DIR_SALIDAS / nombre).write_text(f"# {asunto}\n\nUnidad: {unidad}\n\n{cuerpo}\n", encoding="utf-8")
    return f"OK: borrador guardado en salidas/{nombre}."
