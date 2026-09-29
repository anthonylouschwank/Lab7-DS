"""Genera reporte_lab7.tex a partir de resultados/resumen.json y figures/ (salidas reales del notebook)."""
import json
import re
from pathlib import Path

import lab7_textos as TX

AUTORES = ["Anthony Lou"]          # agrega aquí a tus compañeros
TITULO = "Perfiles y estimación del salario mensual de trabajadores asalariados con Spark MLlib"
CURSO = "CC3066 Data Science — Laboratorio 7"
INST = "Universidad del Valle de Guatemala"
PERIODO = "Semestre II, 2026"

R = json.loads(Path("resultados/resumen.json").read_text(encoding="utf-8"))
FIG = Path("figures")

REEMPL = {"NUM_PERSONA": "identificador del individuo", "NUM_HOGAR": "identificador del hogar"}
ESP = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
       "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
UNI = {"²": r"\textsuperscript{2}", "≤": r"$\leq$", "≥": r"$\geq$", "−": "-", "×": r"$\times$", "±": r"$\pm$",
       "Δ": r"$\Delta$"}


def esc(s):
    s = str(s)
    for a, b in REEMPL.items():
        s = s.replace(a, b)
    s = "".join(ESP.get(ch, ch) for ch in s)
    for a, b in UNI.items():
        s = s.replace(a, b)
    return s


def par(texto):
    return "\n\n".join(esc(p) for p in texto.split("\n\n")) + "\n\n"


def num(x, d=2):
    return f"{x:,.{d}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else esc(x)


def tabla(enc, filas, cap, lab, alin=None):
    alin = alin or "l" + "r" * (len(enc) - 1)
    cuerpo = "\n".join(" & ".join(c if c.startswith("\\") else esc(c) for c in f) + r" \\" for f in filas)
    return (f"\\begin{{table}}[H]\n\\centering\n\\caption{{{esc(cap)}}}\n\\label{{tab:{lab}}}\n"
            f"\\begin{{adjustbox}}{{max width=\\linewidth}}\n\\begin{{tabular}}{{{alin}}}\n\\toprule\n"
            + " & ".join(f"\\textbf{{{esc(e)}}}" for e in enc) + " \\\\\n\\midrule\n" + cuerpo
            + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n\n")


def figura(nombre, cap):
    ruta = FIG / f"{nombre}.png"
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}: ejecuta el notebook completo antes de generar el reporte")
    return (f"\\begin{{figure}}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{{{ruta.as_posix()}}}\n"
            f"\\caption{{{esc(cap)}}}\n\\label{{fig:{nombre}}}\n\\end{{figure}}\n\n")


def ref(tipo, lab):
    return f"{'la Figura' if tipo == 'fig' else 'la Tabla'}~\\ref{{{tipo}:{lab}}}"


def Ref(tipo, lab):
    r = ref(tipo, lab)
    return r[0].upper() + r[1:]


def seccion(titulo):
    return f"\\section{{{esc(titulo)}}}\n\n"


m, v, c, d, B = R["metricas_test"], R["metricas_val"], R["config"], R["dist"], R["bandas"]
ETQ = {"base": "Baseline", "lr": "Regresión lineal", "rf": "Random Forest"}
q, nn, pc = TX.q, TX.nn, TX.pc
S = []

S.append(seccion("Introducción y contexto") + par(
    "Este informe documenta el Laboratorio 7, en el que se analiza el salario mensual de los trabajadores asalariados "
    "de Guatemala con microdatos de la Encuesta Nacional de Empleo e Ingresos Continua (ENEIC) del Instituto Nacional "
    "de Estadística. Todo el procesamiento se realiza con Apache Spark 3.5 y la biblioteca pyspark.ml: carga y "
    "armonización de archivos trimestrales, análisis exploratorio, correlaciones, segmentación con KMeans y modelos "
    "supervisados de regresión lineal y Random Forest. La evaluación sigue un diseño temporal: los trimestres de 2025 "
    "se usan para entrenar y seleccionar modelos, y el primer trimestre de 2026 se reserva como prueba final."))

S.append(seccion("Objetivos y preguntas del análisis") + par(
    "El análisis responde dos preguntas centrales. La primera es qué perfiles de trabajadores asalariados pueden "
    "identificarse según edad, antigüedad en el empleo y jornada habitual. La segunda es qué tan bien puede estimarse el "
    "salario mensual a partir de seis características personales y laborales observadas. Como objetivos específicos se "
    "busca construir una población analítica reproducible, describir la distribución salarial y sus diferencias por "
    "grupo, comparar un modelo lineal regularizado con un modelo de árboles frente a un modelo de referencia y "
    "caracterizar los errores de predicción, en especial en los salarios altos."))

tr = R["eda"]["trimestres"]
S.append(seccion("Descripción de datos y periodos") + par(
    "Se usan las bases de Personas de la ENEIC de los cuatro trimestres de 2025 y del primero de 2026. Cada archivo se "
    "identifica por su procedencia (periodo_archivo) y no por la columna TRIMESTRE del cuestionario.")
    + f"{Ref('tab', 'periodos')} resume el tamaño de la población analítica y el salario por periodo. "
    + par(f"La muestra analítica es estable entre trimestres y el salario nominal aumenta de forma gradual: la mediana "
          f"pasa de {q(tr[0]['mediana'])} en {tr[0]['periodo']} a {q(tr[-1]['mediana'])} en {tr[-1]['periodo']}.")
    + tabla(["Periodo", "Registros analíticos", "Salario medio (Q)", "Salario mediano (Q)"],
            [[t["periodo"], nn(t["n"]), num(t["media"]), num(t["mediana"])] for t in tr],
            "Población analítica y salario por periodo", "periodos"))

P = R["poblacion"]
MOT = list(P["exclusiones"][0]["pasos"])
S.append(seccion("Definición de la población analítica") + par(TX.txt_calidad(R))
         + f"{Ref('tab', 'exclusiones')} detalla las exclusiones por criterio, aplicadas en orden y asignando "
           "cada registro al primer criterio que incumple.\n\n"
         + tabla(["Conjunto", "Original"] + [TX.sin_num(x) for x in MOT] + ["Analítica"],
                 [[e["conjunto"], nn(e["antes"])] + [nn(e["pasos"][x]) for x in MOT] + [nn(e["despues"])]
                  for e in P["exclusiones"]], "Registros excluidos por criterio", "exclusiones"))

S.append(seccion("Variables utilizadas") + par(
    "La variable objetivo es salario_mensual, en quetzales, sin transformación ni recorte. Los seis predictores "
    "obligatorios son tres numéricos (edad, antigüedad en años y horas semanales habituales) y tres categóricos (nivel "
    "educativo, categoría ocupacional y dominio de estudio). Como claves de auditoría se usan el periodo del archivo y "
    "los identificadores del hogar y del individuo. FACTOR, el factor de expansión, se conserva solo con fines "
    "descriptivos. La etiqueta de cluster no se usa como predictor.")
    + tabla(["Variable", "Tipo", "Rol"],
            [["salario_mensual", "numérica (Q)", "objetivo"], ["edad", "numérica (años)", "predictor"],
             ["antiguedad", "numérica (años)", "predictor"], ["horas_semanales", "numérica (horas)", "predictor"],
             ["nivel_educativo", "categórica", "predictor"], ["categoria_ocupacional", "categórica", "predictor"],
             ["dominio", "categórica", "predictor"], ["FACTOR", "numérica", "descriptiva, no usada en modelos"],
             ["periodo_archivo", "texto", "procedencia y clave de auditoría"]],
            "Variables del análisis", "variables", "lll"))

S.append(seccion("Preparación y control de calidad") + par(
    "Cada archivo de Excel se leyó por separado seleccionando las columnas por nombre, porque el archivo del cuarto "
    "trimestre de 2025 tiene 302 columnas en lugar de 270 y un apilamiento por posición mezclaría variables. Los códigos "
    "categóricos se normalizaron a texto, las variables numéricas a double y cada archivo se guardó en Parquet; los "
    "DataFrames se unieron con unionByName. Los códigos se validaron contra los diccionarios oficiales y los valores no "
    "reconocidos se etiquetaron como DESCONOCIDO. Se cuantificaron los faltantes antes de filtrar, se verificó la "
    "unicidad de la clave periodo, hogar e individuo sin eliminar registros y los conjuntos preparados de 2025 y 2026T1 "
    "se guardaron en Parquet por separado.")
    + par(f"Los modelos supervisados se construyen como Pipeline de pyspark.ml. La regresión lineal encadena "
          f"{TX.lista(c['etapas_lr'])}; los indexadores y el codificador usan handleInvalid = keep para no perder filas "
          "con categorías no vistas, y la estandarización se realiza dentro del estimador. Random Forest usa "
          f"{TX.lista(c['etapas_rf'])}, sin estandarización porque los árboles no dependen de la escala."))

d25 = d["2025"]
S.append(seccion("Análisis exploratorio") + par(TX.txt_exploracion(R))
         + f"{Ref('tab', 'distribucion')} compara la distribución del salario en 2025, en el trimestre de "
           "validación y en la prueba. En los tres conjuntos la media supera a la mediana y la asimetría es alta, lo que "
           "anticipa un RMSE mucho mayor que el MAE.\n\n"
         + tabla(["Conjunto", "n", "Media", "Mediana", "p95", "p99", "Máximo", "Asimetría"],
                 [[k, nn(x["n"]), num(x["media"]), num(x["mediana"]), num(x["p95"]), num(x["p99"]), num(x["max"]),
                   num(x["asimetria"])] for k, x in d.items()], "Distribución del salario mensual (Q)", "distribucion")
         + f"En {ref('fig', 'distribucion_categoricas')} se observa la composición de la población analítica de 2025 "
           "por categoría ocupacional, nivel educativo y dominio.\n\n"
         + figura("distribucion_categoricas", "Distribución de registros por categoría ocupacional, nivel educativo y dominio (2025)")
         + f"{Ref('fig', 'distribucion_salario')} muestra la forma de la distribución salarial en escala "
           "lineal y logarítmica: la cola derecha es larga y la escala logarítmica concentra la masa alrededor de la mediana.\n\n"
         + figura("distribucion_salario", "Distribución del salario mensual en 2025, escala lineal y logarítmica")
         + f"{Ref('fig', 'salario_educacion_categoria')} presenta la mediana y el rango intercuartílico "
           "del salario por nivel educativo y por categoría ocupacional.\n\n"
         + figura("salario_educacion_categoria", "Salario mensual por nivel educativo y por categoría ocupacional (2025)")
         + f"{Ref('fig', 'muestra_salario_periodo')} resume la evolución trimestral del tamaño de muestra "
           "y del salario medio y mediano.\n\n"
         + figura("muestra_salario_periodo", "Registros analíticos y salario por periodo"))

nums = list(R["corr"]["pearson_salario"])
mat = {r["variable"]: r for r in R["corr"]["matriz"]}
S.append(seccion("Correlaciones") + par(TX.txt_relaciones(R))
         + f"{Ref('tab', 'correlacion')} contiene la matriz de Pearson calculada con Correlation.corr sobre "
           f"todos los registros de 2025 y {ref('fig', 'correlaciones')} la representa como mapa de calor junto con "
           "la correlación de Spearman.\n\n"
         + tabla(["Variable"] + list(mat), [[k] + [num(r[c], 3) for c in mat] for k, r in mat.items()],
                 "Matriz de correlación de Pearson (2025)", "correlacion")
         + figura("correlaciones", "Mapas de calor de correlación de Pearson y Spearman (2025)"))

K = R["cluster"]
S.append(seccion("Segmentación con KMeans") + par(TX.txt_segmentacion(R))
         + f"{Ref('tab', 'k')} y {ref('fig', 'seleccion_k')} resumen los criterios de elección de K; "
           f"{ref('tab', 'perfiles')} describe los perfiles en unidades originales.\n\n"
         + tabla(["K", "Silueta", "Cluster más pequeño (%)"],
                 [[str(r["K"]), num(r["silueta"], 3), num(r["cluster_min"], 1)] for r in K["tabla_k"]],
                 "Criterios de selección de K (conjunto elegido)", "k")
         + figura("seleccion_k", "Silueta, inercia relativa y tamaño del cluster más pequeño por K")
         + tabla(["Segmento", "n", "%", "Edad media", "Antigüedad media", "Horas medias", "Salario mediano"],
                 [[p["segmento"], nn(p["n"]), num(p["pct"], 1), num(p["edad"], 1), num(p["antiguedad"], 1),
                   num(p["horas"], 1), num(p["salario_mediano"])] for p in K["perfiles"]],
                 f"Perfiles de los {K['k']} segmentos (el salario es descriptivo)", "perfiles")
         + f"{Ref('fig', 'segmentos_dispersion')} muestra los segmentos sobre una muestra para dibujar y "
           f"{ref('fig', 'salario_por_segmento')} la distribución salarial de cada uno con todos sus registros.\n\n"
         + figura("segmentos_dispersion", "Segmentos KMeans en los planos edad, antigüedad y horas (muestra para dibujar)")
         + figura("salario_por_segmento", "Distribución del salario mensual por segmento (descriptiva)"))

sp = {r["conjunto"]: r["filas"] for r in R["splits"]}
tr_n, va_n, te_n = list(sp.values())
S.append(seccion("Diseño de entrenamiento, validación y prueba") + par(
    f"La división es temporal y no aleatoria. El entrenamiento usa 2025T1 a 2025T3 ({nn(tr_n)} registros), la "
    f"validación 2025T4 ({nn(va_n)}) y la prueba 2026T1 ({nn(te_n)}). Los hiperparámetros se eligen por el menor RMSE "
    "de validación; después, los pipelines elegidos se reajustan desde cero con todo 2025 y se evalúan una sola vez en "
    "2026T1. Ningún indexador, codificador ni estimador se ajusta con datos de 2026 y ningún hiperparámetro se modifica "
    "después de observar la prueba."))

S.append(seccion("Modelo de referencia") + par(
    f"El modelo de referencia predice una constante igual a la media del salario de entrenamiento. En la etapa de "
    f"validación esa media se calcula con 2025T1 a 2025T3 ({q(c['media_train_val'])}) y obtiene MAE de "
    f"{q(v['base']['MAE'])}, RMSE de {q(v['base']['RMSE'])} y R² de {v['base']['R2']:.3f}. Para la evaluación final se "
    f"recalcula con todo 2025 ({q(R['media_train'])}). No usa predictores y sirve como piso de comparación: un R² "
    "cercano a cero es el comportamiento esperado de un predictor constante."))

g = R["grids"]
S.append(seccion("Regresión lineal") + par(
    f"Se evaluaron {len(g['lr'])} configuraciones de regularización que combinan regParam y elasticNetParam (ridge, "
    f"elastic net y lasso), seleccionadas por RMSE de validación. La mejor configuración es {c['lr']['nombre']}; las "
    "diferencias entre configuraciones son mínimas, de modo que la regularización apenas modifica el ajuste en este rango.")
    + f"{Ref('tab', 'grid_lr')} ordena los resultados por RMSE de validación.\n\n"
    + tabla(["Configuración", "regParam", "elasticNetParam", "MAE", "RMSE", "R²"],
            [[r["configuracion"], num(r["regParam"]), num(r["elasticNetParam"]), num(r["MAE"]), num(r["RMSE"]),
              num(r["R2"], 4)] for r in g["lr"]], "Regresión lineal en validación 2025T4", "grid_lr"))

S.append(seccion("Random Forest") + par(
    f"Se evaluaron {len(g['rf'])} configuraciones que varían el número de árboles y la profundidad máxima, con semilla "
    f"fija {c['rf']['seed']}. La mejor configuración es {c['rf']['nombre']} ({c['rf']['numTrees']} árboles, profundidad "
    f"{c['rf']['maxDepth']}); los árboles más profundos reducen el error de forma clara, consistente con interacciones "
    "entre educación y categoría ocupacional que el modelo aditivo no representa.")
    + f"{Ref('tab', 'grid_rf')} ordena los resultados por RMSE de validación.\n\n"
    + tabla(["Configuración", "numTrees", "maxDepth", "MAE", "RMSE", "R²"],
            [[r["configuracion"], nn(r["numTrees"]), nn(r["maxDepth"]), num(r["MAE"]), num(r["RMSE"]), num(r["R2"], 4)]
             for r in g["rf"]], "Random Forest en validación 2025T4", "grid_rf"))

kv = TX.mejor(v)
S.append(seccion("Comparación en validación") + f"{Ref('tab', 'val')} " + par(
    f"reúne los tres modelos en 2025T4. {TX.cap(TX.NOM[kv])} obtiene el menor RMSE ({q(v[kv]['RMSE'])}) y ambos "
    f"modelos superan con claridad al de referencia: la regresión lineal reduce el RMSE en "
    f"{q(v['base']['RMSE'] - v['lr']['RMSE'])} y Random Forest en {q(v['base']['RMSE'] - v['rf']['RMSE'])}. En todos "
    "los casos el RMSE es claramente mayor que el MAE por el peso de la cola salarial.")
    + tabla(["Modelo", "MAE", "RMSE", "R²"],
            [[ETQ[k], num(v[k]["MAE"]), num(v[k]["RMSE"]), num(v[k]["R2"], 4)] for k in ETQ],
            "Comparación en validación 2025T4", "val"))

S.append(seccion("Reentrenamiento final") + par(
    f"Con los hiperparámetros fijados en validación, ambos pipelines se reajustaron desde cero con los {nn(d25['n'])} "
    "registros de 2025 y se guardaron en rutas distintas a las de validación. Se verificó que los modelos guardados se "
    "recargan y reproducen exactamente las predicciones. El conjunto de 2026T1 permaneció reservado hasta este punto."))

cn = R["conteos"]
S.append(seccion("Evaluación en 2026T1") + par(TX.interp_conteos(R))
         + f"{Ref('tab', 'test')} presenta las métricas finales calculadas sobre todos los registros de prueba.\n\n"
         + tabla(["Modelo", "MAE 2026", "RMSE 2026", "R² 2026"],
                 [[ETQ[k], num(m[k]["MAE"]), num(m[k]["RMSE"]), num(m[k]["R2"], 4)] for k in ETQ],
                 "Métricas finales en 2026T1 (todos los registros)", "test")
         + par(TX.interp_metricas(R)))

S.append(seccion("Visualización y análisis de errores") + par(
    "El residuo se define como salario observado menos salario predicho: un residuo positivo indica subestimación y uno "
    "negativo sobreestimación. Las gráficas usan una misma muestra determinista de hasta 5,000 registros para ambos "
    "modelos; cada figura incluye un panel a escala completa, que conserva la cola, y uno ampliado.")
    + f"{Ref('fig', 'real_vs_pred_lr')} compara el salario real y el predicho por la regresión lineal.\n\n"
    + figura("real_vs_pred_lr", "Regresión lineal: salario real frente a predicho con la línea y = x (2026T1)")
    + par(TX.interp_scatter(R, "lr"))
    + f"{Ref('fig', 'real_vs_pred_rf')} muestra la misma comparación para Random Forest.\n\n"
    + figura("real_vs_pred_rf", "Random Forest: salario real frente a predicho con la línea y = x (2026T1)")
    + par(TX.interp_scatter(R, "rf"))
    + f"{Ref('fig', 'residuos_lr')} presenta los residuos de la regresión lineal frente al salario predicho.\n\n"
    + figura("residuos_lr", "Regresión lineal: residuos frente al salario predicho (2026T1)")
    + par(TX.interp_residuos(R, "lr"))
    + f"{Ref('fig', 'residuos_rf')} presenta los residuos de Random Forest.\n\n"
    + figura("residuos_rf", "Random Forest: residuos frente al salario predicho (2026T1)")
    + par(TX.interp_residuos(R, "rf")))


def seccion_grupo(titulo, clave, lab, etiqueta, fig, cap_fig):
    G = R[clave]
    return (seccion(titulo) + f"{Ref('tab', lab)} y {ref('fig', fig)} resumen el error por {etiqueta} "
            "con todos los registros de 2026T1.\n\n"
            + tabla(["Grupo", "n", "Salario mediano", "MAE LR", "ME LR", "MAE RF", "ME RF"],
                    [[x["grupo"], nn(x["n"]), num(x["salario_mediano"]), num(x["MAE_lr"]), num(x["ME_lr"]),
                      num(x["MAE_rf"]), num(x["ME_rf"])] for x in G], f"Error por {etiqueta} en 2026T1", lab)
            + figura(fig, cap_fig) + par(TX.interp_grupo(R, clave, etiqueta)))


S.append(seccion_grupo("Errores por nivel educativo", "educ", "educ", "nivel educativo", "error_nivel_educativo",
                       "MAE y error medio por nivel educativo, regresión lineal y Random Forest (2026T1)"))
S.append(seccion_grupo("Errores por dominio", "dominio", "dominio", "dominio", "error_dominio",
                       "MAE y error medio por dominio, regresión lineal y Random Forest (2026T1)"))

p = R["percentiles"]
S.append(seccion("Análisis por percentiles y salarios altos")
         + f"{Ref('tab', 'bandas')} y {ref('fig', 'error_percentiles')} presentan el error por banda "
           "salarial definida con los percentiles de 2026T1.\n\n"
         + tabla(["Banda", "n", "Salario mediano", "MAE LR", "RMSE LR", "ME LR", "MAE RF", "RMSE RF", "ME RF",
                  "% error cuadrático RF"],
                 [[x["grupo"], nn(x["n"]), num(x["salario_mediano"]), num(x["MAE_lr"]), num(x["RMSE_lr"]),
                   num(x["ME_lr"]), num(x["MAE_rf"]), num(x["RMSE_rf"]), num(x["ME_rf"]), pc(x["pct_SSE_rf"])]
                  for x in B], "Error por banda salarial en 2026T1", "bandas")
         + figura("error_percentiles", "MAE y error medio por banda de percentiles del salario (2026T1)")
         + par(TX.interp_bandas(R)))

S.append(seccion("Discusión integral") + f"{Ref('tab', 'valtest')} " + par(
    "reúne las métricas de validación y de prueba de los dos modelos finales.")
    + tabla(["Modelo", "MAE val", "RMSE val", "R² val", "MAE prueba", "RMSE prueba", "R² prueba"],
            [[ETQ[k], num(v[k]["MAE"]), num(v[k]["RMSE"]), num(v[k]["R2"], 4), num(m[k]["MAE"]), num(m[k]["RMSE"]),
              num(m[k]["R2"], 4)] for k in ("lr", "rf")], "Validación 2025T4 frente a prueba 2026T1", "valtest")
    + par(TX.interp_val_test(R)) + par(TX.txt_modelado(R)) + par(TX.txt_errores(R))
    + par("Todos los resultados de este informe son no ponderados: describen los registros analizados y no la "
          "población nacional, para la cual sería necesario ponderar por FACTOR y considerar el diseño muestral."))

S.append(seccion("Limitaciones") + par(TX.txt_limitaciones(R)))
S.append(seccion("Conclusiones") + par(TX.txt_respuestas(R)))

PRE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\IfFileExists{lmodern.sty}{\usepackage{lmodern}}{}
\IfFileExists{spanish.ldf}{\usepackage[spanish,es-nodecimaldot,es-noshorthands,es-tabla]{babel}}{%
  \usepackage[english]{babel}\addto\captionsenglish{\renewcommand{\figurename}{Figura}%
  \renewcommand{\tablename}{Tabla}\renewcommand{\contentsname}{Índice}}}
\usepackage[margin=2.3cm]{geometry}
\usepackage{booktabs,graphicx,float,adjustbox,caption}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{0.5em}
\begin{document}
\begin{titlepage}
\centering
{\large """ + esc(INST) + r"""\par}
\vspace{0.4cm}
{\large """ + esc(CURSO) + r"""\par}
\vspace{3cm}
{\LARGE\bfseries """ + esc(TITULO) + r"""\par}
\vspace{2cm}
{\large """ + r" \\ ".join(esc(a) for a in AUTORES) + r"""\par}
\vfill
{\large """ + esc(PERIODO) + r"""\par}
\end{titlepage}
\tableofcontents
\newpage
"""
tex = PRE + "".join(S) + "\\end{document}\n"
PROHIBIDO = re.compile(r"\b(parte|partes|persona|personas|md|prompt|IA|pendiente|integrante)\b")
assert not PROHIBIDO.search(tex), f"Texto prohibido en el reporte: {PROHIBIDO.search(tex).group()}"
Path("reporte_lab7.tex").write_text(tex, encoding="utf-8")
print("reporte_lab7.tex generado")
