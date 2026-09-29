"""Valida Lab7_SparkMLlib.ipynb ya ejecutado y reporte_lab7.tex. Sale con código 1 si algo falla."""
import json
import re
import sys
from pathlib import Path

nb = json.loads(Path("Lab7_SparkMLlib.ipynb").read_text(encoding="utf-8"))
src = lambda c: "".join(c["source"])
code = [c for c in nb["cells"] if c["cell_type"] == "code" and src(c).strip()]
fallas, avisos = [], []

cuentas = [c.get("execution_count") for c in code]
if cuentas != list(range(1, len(code) + 1)):
    fallas.append("Las celdas no se ejecutaron en orden desde un kernel limpio")
for i, c in enumerate(code):
    for o in c.get("outputs", []):
        if o["output_type"] == "error":
            fallas.append(f"Excepción en celda de código {i + 1}: {o['ename']}: {o['evalue'][:120]}")

textos = [src(c) for c in nb["cells"] if c["cell_type"] == "markdown"]
for c in code:
    for o in c.get("outputs", []):
        textos.append("".join(o.get("data", {}).get("text/markdown", "")))
PROHIB = re.compile(r"\b(persona|integrante|parte|md)\s*\d+\b|\bpendiente\b|insertar aquí|resultado esperado", re.I)
for t in textos:
    for mt in PROHIB.finditer(t):
        fallas.append(f"Texto no permitido en markdown: '{mt.group()}'")

RUTAS = re.compile(r"[A-Za-z]:\\\\|/Users/|/mnt/c/|/home/(?!jovyan)")
for c in code:
    if RUTAS.search(src(c)):
        fallas.append(f"Ruta absoluta del host en código: {RUTAS.search(src(c)).group()}")

imgs = sum("image/png" in o.get("data", {}) for c in code for o in c.get("outputs", []))
FIGS = ["real_vs_pred_lr", "real_vs_pred_rf", "residuos_lr", "residuos_rf", "error_nivel_educativo", "error_dominio",
        "error_percentiles", "distribucion_salario", "correlaciones", "seleccion_k"]
faltan = [f for f in FIGS if not Path(f"figures/{f}.png").exists()]
if faltan:
    fallas.append(f"Figuras no exportadas: {faltan}")

todo = "\n".join(src(c) for c in code)
RUBRICA = {
    "unionByName": r"unionByName", "printSchema": r"printSchema", "cinco registros": r"\.show\(\s*5",
    "Parquet 2025/2026": r"eneic_2025_preparado.*\n?.*eneic_2026T1_preparado|eneic_2026T1_preparado",
    "Correlation.corr": r"Correlation\.corr", "KMeans K 2-5": r"KS = \[2, 3, 4, 5\]", "StandardScaler": r"StandardScaler",
    "StringIndexer": r"StringIndexer", "OneHotEncoder": r"OneHotEncoder", "VectorAssembler": r"VectorAssembler",
    "LinearRegression": r"LinearRegression\(", "RandomForestRegressor": r"RandomForestRegressor\(",
    "semilla fija": r"SEED = 42", "modelos guardados": r"\.write\(\)\.overwrite\(\)\.save",
    "misma muestra común": r"xxhash64", "percentiles": r"percentile\(", "resumen exportado": r"resumen\.json",
}
for nombre, patron in RUBRICA.items():
    if not re.search(patron, todo):
        fallas.append(f"Rúbrica sin evidencia en código: {nombre}")

tex = Path("reporte_lab7.tex")
if tex.exists():
    t = tex.read_text(encoding="utf-8")
    m = re.search(r"\b(parte|partes|persona|personas|md|prompt|IA|pendiente|GitHub)\b", t)
    if m:
        fallas.append(f"Texto no permitido en el reporte: '{m.group()}'")
    for ruta in re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", t):
        if not Path(ruta).exists():
            fallas.append(f"Figura inexistente en el reporte: {ruta}")
else:
    avisos.append("reporte_lab7.tex aún no existe")
log = Path("reporte_lab7.log")
if log.exists():
    lt = log.read_text(errors="ignore")
    for patron in (r"undefined references", r"Reference .* undefined", r"Overfull \\hbox", r"^!"):
        if re.search(patron, lt, re.M):
            fallas.append(f"Compilación LaTeX con problemas: {patron}")

print(f"Celdas de código ejecutadas: {len(code)} | imágenes en salidas: {imgs} | figuras exportadas: {len(list(Path('figures').glob('*.png')))}")
for a in avisos:
    print("AVISO:", a)
if fallas:
    print("\n".join("FALLA: " + f for f in fallas))
    sys.exit(1)
print("OK: notebook y reporte cumplen todas las verificaciones")
