"""Pruebas de la lógica determinista (multas, morosidad, escritura con confirmación).
Ejecutar:  python tests/test_servicios.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agente import servicios as s  # noqa: E402

# Aislar efectos de escritura en una carpeta temporal
tmp = Path(tempfile.mkdtemp())
s.RUTA_DB = tmp / "solicitudes.db"
s.DIR_SALIDAS = tmp / "salidas"


def test_escala_ruido():
    assert "AMONESTACIÓN" in s.calcular_multa("ruido", "A-101")      # 1ª infracción
    assert "0.3 UF" in s.calcular_multa("ruido", "A-102")            # 2ª
    assert "0.6 UF" in s.calcular_multa("ruido", "B-201")            # 3ª
    assert "1.0 UF" in s.calcular_multa("ruido", "B-202")            # 4ª y siguientes


def test_multas_fijas_y_errores():
    assert "0.5 UF" in s.calcular_multa("mudanza", "A-101")
    assert "0.6 UF" in s.calcular_multa("mascotas_excrementos", "B-201")  # reincidente
    assert s.calcular_multa("inexistente", "A-101").startswith("ERROR")
    assert s.calcular_multa("ruido", "Z-999").startswith("ERROR")


def test_morosidad():
    assert "al día" in s.consultar_cuenta("A-101")
    r = s.consultar_cuenta("B-201")
    assert "Art. 15" in r and "Art. 18" in r                          # >60 días y >=3 cuotas
    assert "Art. 18" not in s.consultar_cuenta("A-102")


def test_escritura_exige_confirmacion():
    assert s.registrar_solicitud("A-102", "mantencion", "Portón", False).startswith("NO EJECUTADO")
    assert s.listar_solicitudes("A-102").startswith("La unidad")      # nada se guardó
    assert s.registrar_solicitud("A-102", "mantencion", "Portón no cierra", True).startswith("OK")
    assert "Portón no cierra" in s.listar_solicitudes("A-102")
    assert s.redactar_comunicado("A-102", "Aviso", "Texto", False).startswith("NO EJECUTADO")
    assert s.redactar_comunicado("A-102", "Aviso", "Texto", True).startswith("OK")


if __name__ == "__main__":
    for nombre, fn in list(globals().items()):
        if nombre.startswith("test_"):
            fn()
            print("OK  ", nombre)
    print("Todas las pruebas pasaron")
