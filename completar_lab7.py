"""Completa Lab7_SparkMLlib.ipynb: exporta figuras, corrige el baseline final y agrega las secciones 12-18.
Idempotente: se puede correr varias veces sin duplicar celdas."""
import uuid
import warnings

import nbformat as nbf

warnings.filterwarnings("ignore", message="Cell is missing an id")
NB = "Lab7_SparkMLlib.ipynb"
TAG = "eval_final"
nb = nbf.read(NB, as_version=4)
for _c in nb.cells:
    _c.setdefault("id", uuid.uuid4().hex[:8])


def celda(snippet, tipo="code"):
    for c in nb.cells:
        if c.cell_type == tipo and snippet in c.source:
            return c
    raise LookupError(f"No encontré la celda con: {snippet!r}")


# 1) Helper para exportar figuras
c = celda('MODELS_DIR = BASE / "models"')
if "def guardar_fig" not in c.source:
    c.source += (
        '\n\nFIG_DIR = BASE / "figures"\nFIG_DIR.mkdir(exist_ok=True)\n\n'
        "def guardar_fig(nombre, fig=None):\n"
        '    (fig or plt.gcf()).savefig(FIG_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")\n'
    )

# 2) Guardar las figuras exploratorias existentes
FIGURAS = [
    ('dist_cat = distribucion(', ["distribucion_categoricas"]),
    ("hist_lin = (", ["distribucion_salario"]),
    ("med_edu = mediana_por(", ["salario_educacion_categoria"]),
    ("por_trim = (analitica", ["muestra_salario_periodo"]),
    ("corr_pearson = pd.DataFrame", ["correlaciones"]),
    ('axes[0].set_title("Silueta', ["seleccion_k"]),
    ("muestra_seg = seg.select", ["segmentos_dispersion", "salario_por_segmento"]),
]
for snippet, nombres in FIGURAS:
    c = celda(snippet)
    if "guardar_fig(" in c.source:
        continue
    partes = c.source.split("plt.show()")
    assert len(partes) - 1 == len(nombres), f"Número de plt.show() inesperado en {snippet}"
    c.source = "".join(p + (f'guardar_fig("{n}")\nplt.show()' if i < len(nombres) else "")
                       for i, (p, n) in enumerate(zip(partes, nombres + [None])))

# 3) Baseline final con la media de todo 2025 (no la de 2025T1-T3)
c = celda("baseline_test = test.withColumn")
if "media_final_2025" not in c.source:
    c.source = c.source.replace(
        'baseline_test = test.withColumn("prediction", F.lit(float(media_train)))',
        "media_final_2025 = train_final.agg(F.mean(LABEL)).first()[0]   # baseline final: media de todo 2025\n"
        'baseline_test = test.withColumn("prediction", F.lit(float(media_final_2025)))')
    c.source = c.source.replace("Baseline (media train 2025T1–T3)", "Baseline (media 2025 completo)")
    assert "media_final_2025" in c.source

c = celda("**Verificación 2026T1.**", "markdown")
c.source = (
    "**Verificación 2026T1.** Ambos modelos transformaron todas las filas elegibles de 2026T1 sin pérdidas "
    '(`handleInvalid="keep"`). Ningún indexador, encoder ni estimador se ajustó con 2026. Para la evaluación final, '
    "el baseline usa la media del conjunto final de entrenamiento (todo 2025), no la de 2025T1–T3. Las predicciones "
    "quedan en `data/processed/predicciones_2026T1/` y la evaluación completa está en la sección 12.")

# 4) Nuevas secciones
M, C = "markdown", "code"
CELDAS = [
(M, r"""
## 12. Evaluación final sobre 2026T1

Los modelos finales (hiperparámetros fijados con 2025T4 y reajustados con todo 2025) se evalúan una sola vez sobre 2026T1. Las métricas y las tablas por grupo usan **todos** los registros de prueba; las gráficas usan una muestra común de hasta 5,000 registros. Las interpretaciones de esta sección se redactan a partir de los valores calculados en cada ejecución.

### 12.1 Mismo conjunto de prueba para ambos modelos
"""),
(C, r"""
import importlib
from IPython.display import Markdown
import lab7_textos as TX
importlib.reload(TX)

RES_DIR = BASE / "resultados"
RES_DIR.mkdir(exist_ok=True)
ETQ = {"base": "Baseline", "lr": "Regresión lineal", "rf": "Random Forest"}
QF = mtick.StrMethodFormatter("{x:,.0f}")
R7 = {}

def md(texto):
    display(Markdown(texto))

def registros(df):
    return json.loads(df.to_json(orient="records", force_ascii=False))

media_final_2025 = train_final.agg(F.mean(LABEL)).first()[0]
pl = pred_lr_2026.select(*CLAVE, F.col("prediction").alias("pred_lr"))
pr = pred_rf_2026.select(*CLAVE, F.col("prediction").alias("pred_rf"))
preds = (test.select(*CLAVE, LABEL, "nivel_educativo", "dominio")
         .join(pl, CLAVE).join(pr, CLAVE)
         .withColumn("pred_base", F.lit(float(media_final_2025)))
         .withColumn("res_base", F.col(LABEL) - F.col("pred_base"))
         .withColumn("res_lr", F.col(LABEL) - F.col("pred_lr"))
         .withColumn("res_rf", F.col(LABEL) - F.col("pred_rf"))
         .cache())

def dif_claves(a, b):
    a, b = a.select(*CLAVE), b.select(*CLAVE)
    return a.exceptAll(b).count() + b.exceptAll(a).count()

cont = {"test": test.count(), "claves_unicas_test": test.select(*CLAVE).distinct().count(),
        "lr": pred_lr_2026.count(), "rf": pred_rf_2026.count(), "union": preds.count(),
        "dif_claves_lr": dif_claves(test, pred_lr_2026), "dif_claves_rf": dif_claves(test, pred_rf_2026)}
display(pd.DataFrame([
    {"verificación": "DataFrame elegible 2026T1", "valor": cont["test"]},
    {"verificación": "Claves distintas en 2026T1", "valor": cont["claves_unicas_test"]},
    {"verificación": "Predicciones regresión lineal", "valor": cont["lr"]},
    {"verificación": "Predicciones Random Forest", "valor": cont["rf"]},
    {"verificación": "Unión de predicciones por claves", "valor": cont["union"]},
    {"verificación": "Claves distintas test vs. LR (ambos sentidos)", "valor": cont["dif_claves_lr"]},
    {"verificación": "Claves distintas test vs. RF (ambos sentidos)", "valor": cont["dif_claves_rf"]},
]))
assert cont["test"] == cont["claves_unicas_test"] == cont["lr"] == cont["rf"] == cont["union"]
assert cont["dif_claves_lr"] == 0 and cont["dif_claves_rf"] == 0
R7["conteos"] = {**cont, "claves": CLAVE}
md(TX.interp_conteos(R7))
"""),
(M, r"""
### 12.2 Métricas finales (MAE, RMSE, R²)

El baseline usa la media de `salario_mensual` del conjunto final de entrenamiento (todo 2025), nunca de la prueba.
"""),
(C, r"""
def metricas(df, k):
    ybar = df.agg(F.mean(LABEL)).first()[0]
    e = F.col(f"res_{k}")
    r = df.agg(F.count(F.lit(1)).alias("n"), F.mean(F.abs(e)).alias("MAE"), F.sqrt(F.mean(e ** 2)).alias("RMSE"),
               F.mean(e).alias("ME"), F.sum(e ** 2).alias("sse"),
               F.sum((F.col(LABEL) - F.lit(ybar)) ** 2).alias("sst")).first()
    return {"n": r["n"], "MAE": r["MAE"], "RMSE": r["RMSE"], "R2": 1 - r["sse"] / r["sst"], "ME": r["ME"]}

def dist(df):
    r = df.agg(F.count(LABEL).alias("n"), F.mean(LABEL).alias("media"),
               F.expr(f"percentile({LABEL}, array(0.5, 0.95, 0.99))").alias("p"),
               F.skewness(LABEL).alias("asimetria"), F.max(LABEL).alias("max")).first()
    return {"n": r["n"], "media": r["media"], "mediana": r["p"][0], "p95": r["p"][1], "p99": r["p"][2],
            "asimetria": r["asimetria"], "max": r["max"]}

R7["media_train"] = media_final_2025
R7["metricas_test"] = {k: metricas(preds, k) for k in ETQ}
for k, met in (("lr", met_lr_2026), ("rf", met_rf_2026)):   # control cruzado con RegressionEvaluator
    assert np.isclose(R7["metricas_test"][k]["RMSE"], met["rmse"]) and np.isclose(R7["metricas_test"][k]["R2"], met["r2"])

fila_val = lambda t: {"MAE": float(t.iloc[0]["MAE"]), "RMSE": float(t.iloc[0]["RMSE"]), "R2": float(t.iloc[0]["R2"])}
R7["metricas_val"] = {"base": {"MAE": met_baseline_val["mae"], "RMSE": met_baseline_val["rmse"], "R2": met_baseline_val["r2"]},
                      "lr": fila_val(tabla_lr), "rf": fila_val(tabla_rf)}
R7["dist"] = {"2025": dist(df25), "2025T4": dist(val), "2026T1": dist(test)}

tabla_2026 = pd.DataFrame([{"Modelo": ETQ[k], "MAE 2026": v["MAE"], "RMSE 2026": v["RMSE"], "R² 2026": v["R2"]}
                           for k, v in R7["metricas_test"].items()])
tabla_2026.to_csv(RES_DIR / "metricas_2026T1.csv", index=False)
display(tabla_2026.style.format({"MAE 2026": "{:,.2f}", "RMSE 2026": "{:,.2f}", "R² 2026": "{:.4f}"}).hide(axis="index"))
print("Distribución del salario por conjunto:")
display(pd.DataFrame(R7["dist"]).T)
md(TX.interp_metricas(R7))
"""),
(M, r"""
## 13. Análisis de errores

**Convención del residuo** (se usa en todas las tablas, gráficas y textos):

$$\text{residuo} = \text{salario\_mensual} - \text{prediction}$$

- residuo **positivo**: el modelo **subestimó** el salario;
- residuo **negativo**: el modelo **sobreestimó** el salario;
- residuo cercano a cero: predicción cercana al valor observado.

El error medio (ME) de las tablas es el promedio de este residuo. La muestra para dibujar es una sola, determinista (hash de las claves con semilla 42), tomada de la tabla conjunta de predicciones y usada en todos los gráficos comparativos. Las métricas se siguen calculando con todos los registros.
"""),
(C, r"""
PCTS = [0.25, 0.5, 0.75, 0.95, 0.99]
pv = [float(x) for x in test.agg(F.expr(f"percentile({LABEL}, array({', '.join(map(str, PCTS))}))")).first()[0]]
R7["percentiles"] = dict(zip(["p25", "p50", "p75", "p95", "p99"], pv))
BANDA = (F.when(F.col(LABEL) <= pv[0], "1. hasta p25").when(F.col(LABEL) <= pv[1], "2. p25–p50")
         .when(F.col(LABEL) <= pv[2], "3. p50–p75").when(F.col(LABEL) <= pv[3], "4. p75–p95")
         .when(F.col(LABEL) <= pv[4], "5. p95–p99").otherwise("6. sobre p99"))

def por_grupo(expr):
    aggs = [F.count(F.lit(1)).alias("n"), F.expr(f"percentile({LABEL}, 0.5)").alias("salario_mediano"),
            F.mean(LABEL).alias("salario_medio")]
    for k in ("lr", "rf"):
        e = F.col(f"res_{k}")
        aggs += [F.mean(F.abs(e)).alias(f"MAE_{k}"), F.sqrt(F.mean(e ** 2)).alias(f"RMSE_{k}"),
                 F.mean(e).alias(f"ME_{k}"), F.sum(e ** 2).alias(f"SSE_{k}")]
    d = preds.groupBy(expr.alias("grupo")).agg(*aggs).toPandas()
    d["pct_n"] = d["n"] / d["n"].sum()
    for k in ("lr", "rf"):
        sse = d.pop(f"SSE_{k}")
        d[f"pct_SSE_{k}"] = sse / sse.sum()
    return d

def stats_residuo(k):
    pc_, rc = f"pred_{k}", f"res_{k}"
    t1, t2 = preds.approxQuantile(pc_, [1 / 3, 2 / 3], 0.0)
    tercil = F.when(F.col(pc_) <= t1, "bajo").when(F.col(pc_) <= t2, "medio").otherwise("alto")
    ter = (preds.groupBy(tercil.alias("tercil")).agg(F.count(F.lit(1)).alias("n"), F.mean(pc_).alias("pred_media"),
                                                     F.mean(rc).alias("me"), F.stddev(rc).alias("sd"))
           .toPandas().set_index("tercil").reindex(["bajo", "medio", "alto"]).dropna().reset_index())
    g = preds.agg(F.mean((F.col(rc) > 0).cast("double")).alias("pct_pos"), F.skewness(rc).alias("asimetria"),
                  F.expr(f"percentile({rc}, array(0.01, 0.99))").alias("p"),
                  F.mean((F.abs(F.col(rc)) <= 0.2 * F.col(LABEL)).cast("double")).alias("pct_20"),
                  F.min(rc).alias("min"), F.max(rc).alias("max")).first()
    return {"terciles": registros(ter), "pct_pos": g["pct_pos"], "asimetria": g["asimetria"], "p01": g["p"][0],
            "p99": g["p"][1], "pct_20": g["pct_20"], "min": g["min"], "max": g["max"]}

bandas = por_grupo(BANDA).sort_values("grupo").reset_index(drop=True)
R7["bandas"] = registros(bandas)
R7["residuos"] = {k: stats_residuo(k) for k in ("lr", "rf")}

muestra = (preds.withColumn("_h", F.xxhash64(*CLAVE, F.lit(SEED))).orderBy("_h", *CLAVE).limit(5000)
           .select(*CLAVE, LABEL, "pred_lr", "pred_rf", "res_lr", "res_rf").toPandas())
assert muestra[CLAVE].drop_duplicates().shape[0] == len(muestra)
print(f"Muestra común para gráficas: {len(muestra):,} de {cont['union']:,} registros (misma muestra para ambos modelos)")

VALS = muestra[[LABEL, "pred_lr", "pred_rf"]].to_numpy()
LO, HI = min(0.0, float(VALS.min())), float(VALS.max())
ZOOM = float(np.percentile(muestra[LABEL], 95))
RES_M = muestra[["res_lr", "res_rf"]].to_numpy()
RMAX = float(np.abs(RES_M).max())
R01, R99 = np.percentile(RES_M, [1, 99])
PMIN = min(0.0, float(muestra[["pred_lr", "pred_rf"]].min().min()))
PMAX = float(muestra[["pred_lr", "pred_rf"]].max().max())

def fig_real_pred(k):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, top, sub in ((axs[0], HI, "escala completa"), (axs[1], ZOOM, "ampliación hasta p95 de la muestra")):
        ax.scatter(muestra[LABEL], muestra[f"pred_{k}"], s=6, alpha=0.25, edgecolors="none", color="#4C72B0")
        ax.plot([LO, top], [LO, top], color="#C44E52", ls="--", lw=1.3, label="y = x")
        ax.set(xlim=(LO, top), ylim=(LO, top), title=sub,
               xlabel="Salario mensual observado (Q)", ylabel="Salario mensual predicho (Q)")
        ax.xaxis.set_major_formatter(QF); ax.yaxis.set_major_formatter(QF); ax.legend(loc="upper left")
    fig.suptitle(f"{ETQ[k]}: salario real vs. predicho — 2026T1, muestra común n = {len(muestra):,}")
    fig.tight_layout()
    guardar_fig(f"real_vs_pred_{k}", fig)
    plt.show()

def fig_residuos(k):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, ylim, sub in ((axs[0], (-RMAX, RMAX), "escala completa"), (axs[1], (R01, R99), "ampliación entre percentiles 1 y 99")):
        ax.scatter(muestra[f"pred_{k}"], muestra[f"res_{k}"], s=6, alpha=0.25, edgecolors="none", color="#55A868")
        ax.axhline(0, color="#C44E52", ls="--", lw=1.3, label="residuo = 0")
        ax.set(xlim=(PMIN, PMAX), ylim=ylim, title=sub,
               xlabel="Salario mensual predicho (Q)", ylabel="Residuo = observado − predicho (Q)")
        ax.xaxis.set_major_formatter(QF); ax.yaxis.set_major_formatter(QF); ax.legend(loc="upper left")
    fig.suptitle(f"{ETQ[k]}: residuos vs. salario predicho — 2026T1, muestra común n = {len(muestra):,}")
    fig.tight_layout()
    guardar_fig(f"residuos_{k}", fig)
    plt.show()

def fig_grupo(d, nombre, titulo):
    d = d.set_index(d["grupo"] + d["n"].map(lambda n: f" (n={n:,})"))
    fig, axs = plt.subplots(1, 2, figsize=(14, max(3.5, 0.55 * len(d) + 1.5)), sharey=True)
    ren = {"MAE_lr": "Regresión lineal", "MAE_rf": "Random Forest", "ME_lr": "Regresión lineal", "ME_rf": "Random Forest"}
    d[["MAE_lr", "MAE_rf"]].rename(columns=ren).iloc[::-1].plot.barh(ax=axs[0], color=["#4C72B0", "#DD8452"])
    d[["ME_lr", "ME_rf"]].rename(columns=ren).iloc[::-1].plot.barh(ax=axs[1], color=["#4C72B0", "#DD8452"])
    axs[1].axvline(0, color="black", lw=0.8)
    axs[0].set(title="MAE", xlabel="Quetzales", ylabel="")
    axs[1].set(title="Error medio (+ subestima, − sobreestima)", xlabel="Quetzales", ylabel="")
    for ax in axs:
        ax.xaxis.set_major_formatter(QF)
    fig.suptitle(titulo)
    fig.tight_layout()
    guardar_fig(nombre, fig)
    plt.show()
"""),
(M, "### 13.1 Salario real frente a salario predicho — regresión lineal"),
(C, r"""
fig_real_pred("lr")
md(TX.interp_scatter(R7, "lr"))
"""),
(M, "### 13.2 Salario real frente a salario predicho — Random Forest"),
(C, r"""
fig_real_pred("rf")
md(TX.interp_scatter(R7, "rf"))
"""),
(M, "### 13.3 Residuos frente al salario predicho — regresión lineal"),
(C, r"""
fig_residuos("lr")
md(TX.interp_residuos(R7, "lr"))
"""),
(M, "### 13.4 Residuos frente al salario predicho — Random Forest"),
(C, r"""
fig_residuos("rf")
md(TX.interp_residuos(R7, "rf"))
"""),
(M, r"""
## 14. Error por nivel educativo

Calculado con todos los registros de 2026T1. Orden: nivel educativo ascendente. ME = error medio del residuo.
"""),
(C, r"""
COLS_GRUPO = ["grupo", "n", "salario_mediano", "MAE_lr", "ME_lr", "MAE_rf", "ME_rf"]
educ = por_grupo(F.col("nivel_educativo"))
educ = educ.set_index("grupo").reindex([g for g in ORDEN_EDU if g in set(educ["grupo"])]).reset_index()
R7["educ"] = registros(educ)
educ.to_csv(RES_DIR / "error_nivel_educativo.csv", index=False)
display(educ[COLS_GRUPO])
fig_grupo(educ, "error_nivel_educativo", "Error por nivel educativo — 2026T1, todos los registros")
md(TX.interp_grupo(R7, "educ", "nivel educativo"))
"""),
(M, "## 15. Error por dominio\n\nCalculado con todos los registros de 2026T1."),
(C, r"""
dominio = por_grupo(F.col("dominio"))
dominio = dominio.set_index("grupo").reindex([g for g in ORDEN_DOM if g in set(dominio["grupo"])]).reset_index()
R7["dominio"] = registros(dominio)
dominio.to_csv(RES_DIR / "error_dominio.csv", index=False)
display(dominio[COLS_GRUPO])
fig_grupo(dominio, "error_dominio", "Error por dominio — 2026T1, todos los registros")
md(TX.interp_grupo(R7, "dominio", "dominio"))
"""),
(M, r"""
## 16. Análisis por percentiles de salario

Percentiles exactos de `salario_mensual` en 2026T1 calculados con Spark. Las bandas usan todos los registros; `pct_SSE` es la proporción del error cuadrático total que aporta cada banda.
"""),
(C, r"""
display(pd.DataFrame([R7["percentiles"]]))
bandas.to_csv(RES_DIR / "error_percentiles.csv", index=False)
display(bandas[["grupo", "n", "salario_mediano", "MAE_lr", "RMSE_lr", "ME_lr", "MAE_rf", "RMSE_rf", "ME_rf",
                "pct_n", "pct_SSE_lr", "pct_SSE_rf"]]
        .style.format({c: "{:.1%}" for c in ("pct_n", "pct_SSE_lr", "pct_SSE_rf")} |
                      {c: "{:,.2f}" for c in ("salario_mediano", "MAE_lr", "RMSE_lr", "ME_lr", "MAE_rf", "RMSE_rf", "ME_rf")})
        .hide(axis="index"))
fig_grupo(bandas, "error_percentiles", "Error por banda salarial (percentiles de 2026T1) — todos los registros")
md(TX.interp_bandas(R7))
"""),
(M, "## 17. Comparación validación (2025T4) vs. prueba (2026T1)"),
(C, r"""
vt = pd.DataFrame([{"Modelo": ETQ[k],
                    **{f"{m} val 2025T4": R7["metricas_val"][k][m] for m in ("MAE", "RMSE", "R2")},
                    **{f"{m} prueba 2026T1": R7["metricas_test"][k][m] for m in ("MAE", "RMSE", "R2")}} for k in ("lr", "rf")])
vt.to_csv(RES_DIR / "validacion_vs_prueba.csv", index=False)
display(vt.style.format({c: ("{:.4f}" if "R2" in c else "{:,.2f}") for c in vt.columns if c != "Modelo"}).hide(axis="index"))
md(TX.interp_val_test(R7))
"""),
(C, r"""
# Los modelos guardados se recargan y reproducen las predicciones
lote = test.orderBy(*CLAVE).limit(500)
for k, ruta, modelo in (("lr", ruta_lr_final, modelo_lr_final), ("rf", ruta_rf_final, modelo_rf_final)):
    a = modelo.transform(lote).select("prediction").toPandas()["prediction"]
    b = PipelineModel.load(str(ruta)).transform(lote).select("prediction").toPandas()["prediction"]
    print(f"{ETQ[k]}: {ruta.relative_to(BASE)} recargado, diferencia máxima = {np.abs(a - b).max():.2e}")
    assert np.allclose(a, b)
"""),
(M, r"""
## 18. Discusión final integral

Esta sección conecta los resultados de todo el laboratorio. Las cifras provienen de las tablas calculadas en esta misma ejecución.
"""),
(C, r"""
exc = []
for fila, nombre in (("Total 2025", "2025"), (PERIODO_TEST, PERIODO_TEST)):
    r = conteo.loc[fila]
    exc.append({"conjunto": nombre, "antes": int(r["antes"]), "despues": int(r["después"]),
                "pasos": {m: int(r[m]) for m in MOTIVOS}})
ap = apariciones.set_index("periodos")["count"]
fac = df25.agg(F.mean("FACTOR"), (F.sum(F.col(LABEL) * F.col("FACTOR")) / F.sum("FACTOR")), F.mean(LABEL)).first()
suma_fac = df25.groupBy("periodo_archivo").agg(F.sum("FACTOR").alias("s")).toPandas()["s"]
R7["poblacion"] = {
    "n_2025": df25.count(), "n_2026": df26.count(), "exclusiones": exc,
    "repetidos": {"ids": int(ap.sum()), "pct_mas_de_uno": float(ap[ap.index > 1].sum() / ap.sum()),
                  "pct_cuatro": float(ap.get(4, 0) / ap.sum())},
    "factor": {"media": fac[0], "media_ponderada": fac[1], "media_no_ponderada": fac[2],
               "suma_min": float(suma_fac.min()), "suma_max": float(suma_fac.max())},
}
R7["eda"] = {
    "trimestres": registros(por_trim.rename_axis("periodo").reset_index()[["periodo", "n", "mediana", "media"]]),
    "educ": registros(med_edu.rename_axis("grupo").reset_index()),
    "cat": registros(med_cat.rename_axis("grupo").reset_index()),
}
nums = [v for v in VARS_NUM if v != LABEL]
R7["corr"] = {"pearson_salario": {v: float(corr_pearson.loc[v, LABEL]) for v in nums},
              "spearman_salario": {v: float(corr_spearman.loc[v, LABEL]) for v in nums},
              "edad_antig": float(corr_pearson.loc["edad", "antiguedad"]),
              "matriz": registros(corr_pearson.rename_axis("variable").reset_index())}
tk = resultados_km[resultados_km["conjunto"] == CONJUNTO_ELEGIDO]
R7["cluster"] = {
    "k": int(K_ELEGIDO), "vars": CONJUNTOS[CONJUNTO_ELEGIDO], "conjunto": CONJUNTO_ELEGIDO,
    "tabla_k": registros(tk[["K", "silueta", "cluster_min_%"]].rename(columns={"cluster_min_%": "cluster_min"})),
    "perfiles": registros(perfil.rename_axis("segmento").reset_index()[["segmento", "n", "%", "edad_media", "antig_media", "horas_media", "salario_mediano*"]]
                          .rename(columns={"%": "pct", "edad_media": "edad", "antig_media": "antiguedad",
                                           "horas_media": "horas", "salario_mediano*": "salario_mediano"})),
}
R7["config"] = {"lr": {k: mejor_lr_cfg[k] for k in ("nombre", "regParam", "elasticNetParam")},
                "rf": {**{k: mejor_rf_cfg[k] for k in ("nombre", "numTrees", "maxDepth")}, "seed": SEED},
                "predictores": PREDICTORES_NUM + PREDICTORES_CAT, "media_train_val": media_train,
                "etapas_lr": [type(s).__name__ for s in modelo_lr_final.stages],
                "etapas_rf": [type(s).__name__ for s in modelo_rf_final.stages]}
R7["grids"] = {"lr": registros(tabla_lr.drop(columns="segundos")), "rf": registros(tabla_rf.drop(columns="segundos"))}
R7["splits"] = registros(splits_tbl)

for titulo, texto in TX.discusion(R7):
    md(f"### {titulo}\n\n{texto}")

with open(RES_DIR / "resumen.json", "w", encoding="utf-8") as fh:
    json.dump(R7, fh, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(f"Resultados exportados a {(RES_DIR / 'resumen.json').relative_to(BASE)} y figuras a {FIG_DIR.relative_to(BASE)}/")
"""),
]

nb.cells = [c for c in nb.cells if TAG not in c.metadata.get("tags", [])]
for tipo, src in CELDAS:
    c = nbf.v4.new_markdown_cell(src.strip()) if tipo == M else nbf.v4.new_code_cell(src.strip())
    c.metadata["tags"] = [TAG]
    nb.cells.append(c)
nbf.write(nb, NB)
print(f"Notebook actualizado: {len(CELDAS)} celdas nuevas, figuras y baseline final corregidos.")
