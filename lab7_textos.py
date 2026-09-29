"""Interpretaciones del Lab 7 redactadas a partir de los resultados ejecutados (notebook y reporte)."""
import re

NOM = {"base": "el modelo de referencia", "lr": "la regresión lineal", "rf": "Random Forest"}
MOD = ("lr", "rf")


def q(x): return f"-Q{abs(x):,.2f}" if x < 0 else f"Q{x:,.2f}"
def pc(x): return f"{100 * x:.1f}%"
def nn(x): return f"{int(round(x)):,}"
def cap(s): return s[:1].upper() + s[1:]
def signo(me): return "subestimación" if me > 0 else "sobreestimación"
def var(a, b): return (b - a) / a if a else float("nan")
def mejor(m, met="RMSE"): return min(MOD, key=lambda k: m[k][met])
def otro(k): return "rf" if k == "lr" else "lr"
def umbral(G): return max(30, 0.01 * sum(g["n"] for g in G))
def sin_num(s): return re.sub(r"^\d+\s+", "", s)


def lista(xs):
    xs = list(xs)
    if not xs:
        return "ninguna categoría"
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " y " + xs[-1]


def _pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx, sy = sum((a - mx) ** 2 for a in x) ** 0.5, sum((b - my) ** 2 for b in y) ** 0.5
    return sxy / (sx * sy) if sx and sy else float("nan")


def _sub_sob(sub, sob):
    if not sob:
        return f"Todas las categorías tienen error medio positivo, es decir, subestimación promedio en {lista(sub)}."
    if not sub:
        return f"Todas las categorías tienen error medio negativo, es decir, sobreestimación promedio."
    return f"Hay subestimación promedio en {lista(sub)} y sobreestimación promedio en {lista(sob)}."


def interp_conteos(R):
    c = R["conteos"]
    return (f"El DataFrame elegible de 2026T1 tiene {nn(c['test'])} registros con {nn(c['claves_unicas_test'])} claves "
            f"distintas; la regresión lineal genera {nn(c['lr'])} predicciones y Random Forest {nn(c['rf'])}. Las claves "
            f"de auditoría ({', '.join(c['claves'])}) coinciden en ambos sentidos y la unión de predicciones conserva "
            f"{nn(c['union'])} registros, por lo que los dos modelos se evalúan sobre exactamente los mismos registros y "
            "ningún caso se descarta por el tamaño de su error.")


def interp_metricas(R):
    m, v, c, d = R["metricas_test"], R["metricas_val"], R["conteos"], R["dist"]
    k, kmae, kv, b = mejor(m), mejor(m, "MAE"), mejor(v), m["base"]
    o = otro(k)
    t = [f"Sobre los {nn(c['test'])} registros de 2026T1, {NOM[k]} presenta el menor RMSE ({q(m[k]['RMSE'])} frente a "
         f"{q(m[o]['RMSE'])} de {NOM[o]})"
         + (" y también el menor MAE" if kmae == k else f", mientras que el menor MAE corresponde a {NOM[kmae]}")
         + f" ({q(m[kmae]['MAE'])}). El modelo de referencia predice la media salarial del entrenamiento final 2025 "
         f"({q(R['media_train'])}) y obtiene MAE de {q(b['MAE'])} y RMSE de {q(b['RMSE'])}; el mejor MAE lo reduce en "
         f"{pc(1 - m[kmae]['MAE'] / b['MAE'])}."]
    t.append(f"En la validación 2025T4 el menor RMSE fue de {NOM[kv]}, por lo que el orden entre algoritmos "
             + ("se mantiene en la prueba." if kv == k else "se invierte en la prueba."))
    det = "; ".join(
        f"{NOM[x]} {'aumenta' if m[x]['RMSE'] > v[x]['RMSE'] else 'reduce'} su RMSE de {q(v[x]['RMSE'])} a "
        f"{q(m[x]['RMSE'])} ({var(v[x]['RMSE'], m[x]['RMSE']):+.1%}), su MAE pasa de {q(v[x]['MAE'])} a "
        f"{q(m[x]['MAE'])} ({var(v[x]['MAE'], m[x]['MAE']):+.1%}) y su R² de {v[x]['R2']:.3f} a {m[x]['R2']:.3f}"
        for x in MOD)
    t.append(f"Respecto de 2025T4, {det}.")
    t.append(f"Como contexto descriptivo, la mediana salarial pasa de {q(d['2025T4']['mediana'])} en 2025T4 a "
             f"{q(d['2026T1']['mediana'])} en 2026T1 ({var(d['2025T4']['mediana'], d['2026T1']['mediana']):+.1%}) y la "
             f"media de {q(d['2025T4']['media'])} a {q(d['2026T1']['media'])} "
             f"({var(d['2025T4']['media'], d['2026T1']['media']):+.1%}); frente a todo 2025 la media de prueba es "
             f"{var(d['2025']['media'], d['2026T1']['media']):+.1%} mayor. Un desplazamiento de la distribución puede "
             "acompañar cambios en el error, pero estos resultados no permiten atribuirle una relación causal.")
    ratio = m[k]["RMSE"] / m[k]["MAE"]
    t.append("El MAE resume el error absoluto típico en quetzales, mientras que el RMSE eleva al cuadrado cada error y "
             f"pondera más los casos extremos. Para {NOM[k]} el RMSE es {ratio:.2f} veces el MAE, "
             + ("señal de que un grupo reducido de errores grandes domina la métrica cuadrática."
                if ratio > 1.5 else "lo que indica errores relativamente homogéneos."))
    t.append(f"El R² de {m[k]['R2']:.3f} indica que {NOM[k]} explica alrededor de {pc(max(m[k]['R2'], 0))} de la "
             "variabilidad del salario mensual de 2026T1 respecto de predecir su media; el resto de la variación no queda "
             f"capturado por los seis predictores. El modelo de referencia obtiene R² de {b['R2']:.3f}, cercano a cero, "
             "porque predice una constante calculada con 2025 y no con el conjunto de prueba.")
    return "\n\n".join(t)


def interp_scatter(R, k):
    p, s, B = R["percentiles"], R["residuos"][k], R["bandas"]
    me = lambda i: B[i][f"ME_{k}"]
    t = (f"La mayor concentración de observaciones está entre p25 y p75 del salario observado ({q(p['p25'])} a "
         f"{q(p['p75'])}). Con {NOM[k]}, {pc(s['pct_20'])} de los registros de 2026T1 tiene una predicción a menos de "
         f"20% del salario observado. El error medio es {q(me(0))} hasta p25 ({signo(me(0))}), {q(me(2))} entre p50 y "
         f"p75 ({signo(me(2))}), {q(me(4))} entre p95 y p99 y {q(me(5))} por encima de p99 ({signo(me(5))}). ")
    if me(0) < 0 and me(5) > 0:
        t += ("El patrón es de regresión hacia el centro: los salarios bajos quedan por encima de la línea y = x "
              "(sobreestimados) y los altos por debajo (subestimados).")
    elif me(5) > 0:
        t += "En la cola alta los puntos quedan por debajo de la línea y = x: el modelo subestima los salarios más altos."
    else:
        t += "En la cola alta no se observa una subestimación sistemática respecto de la línea y = x."
    if B[-1][f"RMSE_{k}"] > 2 * B[0][f"RMSE_{k}"]:
        t += (f" La dispersión alrededor de la diagonal crece con el salario: el RMSE pasa de {q(B[0][f'RMSE_{k}'])} "
              f"hasta p25 a {q(B[-1][f'RMSE_{k}'])} sobre p99.")
    return t


def interp_residuos(R, k):
    s, me = R["residuos"][k], R["metricas_test"][k]["ME"]
    T3, a = s["terciles"], s["asimetria"]
    t = (f"{pc(s['pct_pos'])} de los residuos es positivo (subestimación) y su coeficiente de asimetría es {a:.2f}, "
         + ("con una cola larga hacia valores positivos: los errores más grandes corresponden a salarios observados muy "
            "superiores a la predicción. " if a > 0.5 else
            "con una cola hacia valores negativos. " if a < -0.5 else "sin una cola dominante marcada. "))
    if len(T3) >= 2:
        ratio = T3[-1]["sd"] / T3[0]["sd"]
        t += (f"Por terciles del salario predicho, la desviación estándar del residuo pasa de {q(T3[0]['sd'])} en el "
              f"tercil bajo a {q(T3[-1]['sd'])} en el alto ({ratio:.1f} veces), "
              + ("lo que describe heterocedasticidad: la dispersión crece con el nivel predicho. " if ratio > 1.5
                 else "sin un aumento fuerte de la dispersión. ")
              + "El error medio por tercil es " + lista([q(x["me"]) for x in T3]) + ". ")
    t += (f"Los percentiles 1 y 99 del residuo son {q(s['p01'])} y {q(s['p99'])}, y el máximo llega a {q(s['max'])}; "
          "estos valores extremos se conservan en el panel a escala completa. En conjunto, el error medio es "
          f"{q(me)}, es decir, {signo(me)} promedio.")
    return t


def interp_grupo(R, clave, etiqueta):
    G = R[clave]
    u = umbral(G)
    grandes = [g for g in G if g["n"] >= u] or G
    chicos = [g for g in G if g["n"] < u]
    t = []
    for k in MOD:
        hi = max(grandes, key=lambda g: g[f"MAE_{k}"])
        lo = min(grandes, key=lambda g: g[f"MAE_{k}"])
        ratio = hi[f"MAE_{k}"] / lo[f"MAE_{k}"]
        sub = [g["grupo"] for g in G if g[f"ME_{k}"] > 0]
        sob = [g["grupo"] for g in G if g[f"ME_{k}"] < 0]
        top = max(grandes, key=lambda g: g[f"pct_SSE_{k}"])
        t.append(f"Con {NOM[k]}, entre las categorías de {etiqueta} con al menos {nn(u)} registros, el mayor MAE está en "
                 f"{hi['grupo']} ({q(hi[f'MAE_{k}'])}, n = {nn(hi['n'])}) y el menor en {lo['grupo']} "
                 f"({q(lo[f'MAE_{k}'])}, n = {nn(lo['n'])}), una razón de {ratio:.1f} veces, "
                 + ("diferencia materialmente importante. " if ratio > 1.5 else "diferencia moderada. ")
                 + f"{top['grupo']} concentra {pc(top[f'pct_SSE_{k}'])} del error cuadrático con {pc(top['pct_n'])} "
                 f"de los registros. " + _sub_sob(sub, sob))
    if len(grandes) >= 3:
        r = _pearson([g["salario_mediano"] for g in grandes], [g["MAE_rf"] for g in grandes])
        t.append(f"Entre categorías, la correlación entre el salario mediano del grupo y el MAE de Random Forest es "
                 f"{r:.2f}: " + ("el error absoluto tiende a ser mayor donde los salarios son más altos y más dispersos."
                                 if r > 0.5 else "el nivel salarial del grupo no explica por sí solo las diferencias de error."))
    if chicos:
        pl = len(chicos) > 1
        t.append(("Las categorías " if pl else "La categoría ") + lista([f"{g['grupo']} (n = {nn(g['n'])})" for g in chicos])
                 + (" tienen" if pl else " tiene") + f" menos de {nn(u)} registros; sus métricas son inestables y se "
                 "interpretan con precaución.")
    else:
        t.append(f"Todas las categorías superan {nn(u)} registros, por lo que las comparaciones tienen un tamaño de "
                 "grupo razonable.")
    return "\n\n".join(t)


def interp_bandas(R):
    B, p = R["bandas"], R["percentiles"]
    t = [f"Los percentiles del salario de 2026T1 son p25 = {q(p['p25'])}, p50 = {q(p['p50'])}, p75 = {q(p['p75'])}, "
         f"p95 = {q(p['p95'])} y p99 = {q(p['p99'])}; las bandas se construyen con esos cortes y usan todos los registros."]
    for k in MOD:
        a, z = B[0][f"MAE_{k}"], B[-1][f"MAE_{k}"]
        top_n = B[4]["pct_n"] + B[5]["pct_n"]
        top_s = B[4][f"pct_SSE_{k}"] + B[5][f"pct_SSE_{k}"]
        t.append(f"Con {NOM[k]} el MAE {'aumenta' if z > a else 'disminuye'} de {q(a)} hasta p25 a {q(z)} por encima "
                 f"de p99. El error medio es {q(B[4][f'ME_{k}'])} entre p95 y p99 y {q(B[5][f'ME_{k}'])} sobre p99, es "
                 f"decir, {signo(B[5][f'ME_{k}'])} en la cola extrema. Los registros sobre p95 son {pc(top_n)} de la "
                 f"prueba pero aportan {pc(top_s)} del error cuadrático total"
                 + (", por lo que los extremos elevan el RMSE global muy por encima del MAE." if top_s > 3 * top_n
                    else "."))
    peso = lambda k, bs: sum(b["n"] * b[f"MAE_{k}"] for b in bs) / sum(b["n"] for b in bs)
    abajo = min(MOD, key=lambda k: peso(k, B[:4]))
    arriba = min(MOD, key=lambda k: peso(k, B[4:]))
    if abajo == arriba:
        t.append(f"Tanto hasta p95 como sobre p95 el menor error absoluto corresponde a {NOM[abajo]} (MAE de "
                 f"{q(peso(abajo, B[:4]))} y {q(peso(abajo, B[4:]))}, frente a {q(peso(otro(abajo), B[:4]))} y "
                 f"{q(peso(otro(abajo), B[4:]))} de {NOM[otro(abajo)]}).")
    else:
        t.append(f"Hasta p95 el menor error absoluto corresponde a {NOM[abajo]} ({q(peso(abajo, B[:4]))}) y sobre p95 a "
                 f"{NOM[arriba]} ({q(peso(arriba, B[4:]))}).")
    t.append("Ninguno de los dos modelos corrige la subestimación de la cola alta; esta se conserva completa en todas las "
             "métricas, porque eliminarla reduciría artificialmente el error." if all(B[5][f"ME_{k}"] > 0 for k in MOD)
             else "La cola alta se conserva completa en todas las métricas, porque eliminarla reduciría artificialmente el error.")
    return "\n\n".join(t)


def interp_val_test(R):
    v, m, d = R["metricas_val"], R["metricas_test"], R["dist"]
    t = [" ".join(f"{cap(NOM[k])}: MAE de {q(v[k]['MAE'])} a {q(m[k]['MAE'])} ({var(v[k]['MAE'], m[k]['MAE']):+.1%}), "
                  f"RMSE de {q(v[k]['RMSE'])} a {q(m[k]['RMSE'])} ({var(v[k]['RMSE'], m[k]['RMSE']):+.1%}) y R² de "
                  f"{v[k]['R2']:.3f} a {m[k]['R2']:.3f}." for k in MOD)]
    estable = all(abs(var(v[k]["RMSE"], m[k]["RMSE"])) < 0.10 and abs(v[k]["R2"] - m[k]["R2"]) < 0.05 for k in MOD)
    t.append("Los cambios son pequeños (menos de 10% en RMSE y menos de 0.05 en R²) en ambos modelos, lo que indica un "
             "desempeño temporalmente estable entre 2025T4 y 2026T1." if estable else
             "Al menos un modelo cambia de forma apreciable entre validación y prueba, por lo que el desempeño no es "
             "completamente estable en el tiempo.")
    t.append(f"Entre 2025T4 y 2026T1 la mediana salarial cambia {var(d['2025T4']['mediana'], d['2026T1']['mediana']):+.1%} "
             f"y la media {var(d['2025T4']['media'], d['2026T1']['media']):+.1%}. Un nivel salarial algo mayor en 2026 "
             "puede acompañar un MAE mayor, mientras que el RMSE depende sobre todo de cuántos salarios extremos contiene "
             "cada trimestre; la comparación es descriptiva.")
    t.append("El diseño es longitudinal: una fracción de los individuos de 2026T1 probablemente también aparece en 2025, "
             "por lo que la prueba no es completamente independiente del entrenamiento y la estabilidad observada puede "
             "ser optimista. Además, la capacidad predictiva no implica causalidad: los modelos capturan asociaciones "
             "observadas, no el efecto de modificar un predictor.")
    return "\n\n".join(t)


def txt_calidad(R):
    P = R["poblacion"]
    t = ("La población analítica está formada por los asalariados ocupados de 15 años o más con salario positivo, antigüedad válida y "
         f"horas en (0, 168]: {nn(P['n_2025'])} registros de 2025 (entrenamiento y validación) y {nn(P['n_2026'])} de "
         "2026T1 (prueba).")
    for e in P["exclusiones"]:
        pos = [f"{nn(n)} por {sin_num(mo)}" for mo, n in e["pasos"].items() if n > 0]
        t += (f" En {e['conjunto']}, de {nn(e['antes'])} registros originales se excluyen {lista(pos)}, y quedan "
              f"{nn(e['despues'])} ({pc(e['despues'] / e['antes'])}).")
    ceros = [sin_num(mo) for mo, n in P["exclusiones"][0]["pasos"].items() if n == 0]
    if ceros:
        t += " Los criterios " + lista(ceros) + " no excluyen ningún registro."
    t += (" Las exclusiones delimitan la población de interés, asalariados con salario observado, y el salario nunca se "
          "imputa ni se recorta.")
    r = P["repetidos"]
    t += (f" De {nn(r['ids'])} combinaciones de hogar e individuo en 2025, {pc(r['pct_mas_de_uno'])} aparece en más de un "
          f"trimestre y {pc(r['pct_cuatro'])} en los cuatro, reflejo del diseño longitudinal con rotación de la ENEIC; "
          "por eso el número de filas no equivale al número de individuos distintos.")
    f = P["factor"]
    t += (f" FACTOR es el factor de expansión: indica cuántos trabajadores de la población representa cada registro "
          f"(media de {f['media']:,.1f} en 2025 y suma trimestral entre {nn(f['suma_min'])} y {nn(f['suma_max'])}). El "
          f"salario medio ponderado de 2025 sería {q(f['media_ponderada'])}, frente a {q(f['media_no_ponderada'])} sin "
          "ponderar. El análisis principal es no ponderado: describe la muestra analítica y no constituye una estimación "
          "oficial para toda Guatemala.")
    return t


def txt_exploracion(R):
    d, E = R["dist"]["2025"], R["eda"]
    t = (f"En 2025 el salario mensual tiene media {q(d['media'])} y mediana {q(d['mediana'])}; la media supera a la "
         f"mediana en {q(d['media'] - d['mediana'])} y el coeficiente de asimetría es {d['asimetria']:.2f}, lo que "
         f"describe una distribución sesgada a la derecha con una cola alta larga (p95 = {q(d['p95'])}, "
         f"p99 = {q(d['p99'])}, máximo = {q(d['max'])}).")
    ed = E["educ"]
    lo, hi = min(ed, key=lambda r: r["mediana"]), max(ed, key=lambda r: r["mediana"])
    mono = all(a["mediana"] <= b["mediana"] for a, b in zip(ed, ed[1:]))
    t += (f" Por nivel educativo, la mediana {'crece de forma monótona con el nivel, ' if mono else ''}desde "
          f"{q(lo['mediana'])} en {lo['grupo']} hasta {q(hi['mediana'])} en {hi['grupo']} (n = {nn(hi['n'])}).")
    ca = sorted(E["cat"], key=lambda r: r["mediana"])
    t += (f" Por categoría ocupacional, la mediana más baja es {q(ca[0]['mediana'])} en {ca[0]['grupo']} y la más alta "
          f"{q(ca[-1]['mediana'])} en {ca[-1]['grupo']}.")
    tr = E["trimestres"]
    t += (f" Entre {tr[0]['periodo']} y {tr[-1]['periodo']} la mediana pasa de {q(tr[0]['mediana'])} a "
          f"{q(tr[-1]['mediana'])} y la media de {q(tr[0]['media'])} a {q(tr[-1]['media'])} en términos nominales.")
    return t


def txt_relaciones(R):
    C = R["corr"]
    cs = sorted(C["pearson_salario"].items(), key=lambda x: -abs(x[1]))
    debil = all(abs(b) < 0.3 for _, b in cs)
    t = ("Las correlaciones de Pearson con el salario mensual en 2025 son " + lista([f"{a} (r = {b:.3f})" for a, b in cs])
         + ("; todas son débiles" if debil else "") + f" y la más fuerte es con {cs[0][0]}. Con Spearman los valores son "
         + lista([f"{a} ({b:.3f})" for a, b in sorted(C["spearman_salario"].items(), key=lambda x: -abs(x[1]))]) + ".")
    r = C["edad_antig"]
    fuera = [abs(row[c]) for row in C["matriz"] for c in row if c not in ("variable", row["variable"])]
    t += f" Edad y antigüedad tienen r = {r:.3f}" + (", la asociación más fuerte de la matriz" if abs(r) >= max(fuera) - 1e-9 else "")
    t += (": los trabajadores de mayor edad tienden a acumular más antigüedad y ambos predictores comparten información."
          if r > 0.3 else ", una asociación débil.")
    t += " Estas correlaciones solo miden asociación y no indican que una variable cause cambios en el salario."
    return t


def txt_segmentacion(R):
    K = R["cluster"]
    sil = {r["K"]: r["silueta"] for r in K["tabla_k"]}
    kmin = {r["K"]: r["cluster_min"] for r in K["tabla_k"]}
    t = (f"La segmentación con KMeans usó {lista(K['vars'])} estandarizadas con StandardScaler y comparó K = 2 a 5. "
         "Las siluetas del conjunto elegido fueron " + lista([f"{sil[k]:.3f} (K = {k})" for k in sorted(sil)]) + ".")
    cand = [k for k in sil if k >= 3 and kmin[k] >= 10]
    if cand and max(cand, key=lambda k: sil[k]) == K["k"]:
        t += (f" Se eligió K = {K['k']} por tener la mayor silueta entre K ≥ 3 con ningún cluster menor a 10% de los "
              f"registros (el más pequeño tiene {kmin[K['k']]:.1f}%) y perfiles interpretables; K = 2 tiene mayor silueta "
              "pero es demasiado grueso para describir perfiles.")
    else:
        t += f" Se eligió K = {K['k']} con un cluster mínimo de {kmin[K['k']]:.1f}% de los registros."
    t += (" El salario no se incluyó en la segmentación: con el salario en quetzales un cluster se dedica a aislar la cola "
          "alta y la silueta baja, y con su logaritmo no hay una ganancia clara; así los perfiles se definen por "
          "condiciones laborales y el salario queda como variable descriptiva. Perfiles: "
          + "; ".join(f"{p['segmento']} ({pc(p['pct'] / 100)}, edad media {p['edad']:.1f} años, antigüedad media "
                      f"{p['antiguedad']:.1f} años, {p['horas']:.1f} horas semanales, salario mediano {q(p['salario_mediano'])})"
                      for p in K["perfiles"]) + ".")
    t += (" Los segmentos son útiles para describir la población, pero se traslapan y dependen de las variables y del "
          "escalamiento elegidos, por lo que son perfiles típicos y no grupos cerrados. La etiqueta de cluster no se usa "
          "como predictor: el modelo supervisado se limita a los seis predictores obligatorios.")
    return t


def txt_modelado(R):
    m, v, c = R["metricas_test"], R["metricas_val"], R["config"]
    return (f"El modelo de referencia predice una constante: la media de entrenamiento ({q(c['media_train_val'])} con "
            f"2025T1–T3 para validación y {q(R['media_train'])} con todo 2025 para la prueba). La mejor regresión lineal "
            f"por RMSE de validación es {c['lr']['nombre']} (regParam = {c['lr']['regParam']}, elasticNetParam = "
            f"{c['lr']['elasticNetParam']}) y el mejor Random Forest es {c['rf']['nombre']} ({c['rf']['numTrees']} "
            f"árboles, profundidad máxima {c['rf']['maxDepth']}, semilla {c['rf']['seed']}). En validación 2025T4, la "
            f"regresión lineal obtuvo MAE {q(v['lr']['MAE'])}, RMSE {q(v['lr']['RMSE'])} y R² {v['lr']['R2']:.3f}, y "
            f"Random Forest MAE {q(v['rf']['MAE'])}, RMSE {q(v['rf']['RMSE'])} y R² {v['rf']['R2']:.3f}. En 2026T1, tras "
            f"reajustar con todo 2025, los valores son MAE {q(m['lr']['MAE'])}, RMSE {q(m['lr']['RMSE'])} y R² "
            f"{m['lr']['R2']:.3f} para la regresión lineal, y MAE {q(m['rf']['MAE'])}, RMSE {q(m['rf']['RMSE'])} y R² "
            f"{m['rf']['R2']:.3f} para Random Forest; {NOM[mejor(m)]} es el algoritmo con menor error en la prueba"
            + (", coherente con su capacidad de capturar interacciones y no linealidades que el modelo lineal aditivo "
               "omite." if mejor(m) == "rf" else "."))


def txt_errores(R):
    B = R["bandas"]
    top = lambda key: max((g for g in R[key] if g["n"] >= umbral(R[key])), key=lambda g: g["MAE_rf"])
    ge, gd = top("educ"), top("dominio")
    return (f"El mayor MAE de Random Forest por nivel educativo aparece en {ge['grupo']} ({q(ge['MAE_rf'])}, "
            f"n = {nn(ge['n'])}) y por dominio en {gd['grupo']} ({q(gd['MAE_rf'])}, n = {nn(gd['n'])}). El error crece con "
            f"el salario: sobre p99 el error medio es {q(B[-1]['ME_lr'])} en la regresión lineal y {q(B[-1]['ME_rf'])} en "
            f"Random Forest ("
            + (f"{signo(B[-1]['ME_lr'])} en ambos" if signo(B[-1]['ME_lr']) == signo(B[-1]['ME_rf'])
               else f"{signo(B[-1]['ME_lr'])} y {signo(B[-1]['ME_rf'])}, respectivamente")
            + f"), mientras que hasta p25 "
            f"es {q(B[0]['ME_lr'])} y {q(B[0]['ME_rf'])}. Los registros sobre p95 concentran "
            f"{pc(B[4]['pct_SSE_rf'] + B[5]['pct_SSE_rf'])} del error cuadrático de Random Forest, de modo que las "
            "observaciones extremas determinan buena proporción del RMSE.")


def txt_limitaciones(R):
    return ("Los datos son observacionales, por lo que ninguna asociación admite interpretación causal. Algunos "
            "individuos se repiten longitudinalmente y las observaciones de periodos distintos pueden ser dependientes, lo "
            "que debilita la independencia entre entrenamiento y prueba. El análisis principal es no ponderado y no usa "
            "FACTOR, por lo que no representa a la población nacional. Los modelos usan exclusivamente seis predictores ("
            + lista(R["config"]["predictores"]) + "), de modo que omiten otros determinantes del salario, como ocupación "
            "específica o rama de actividad. Finalmente, las predicciones describen salarios asociados a perfiles "
            "observados y no deben interpretarse como una recomendación de cuánto debería ganar un trabajador.")


def txt_respuestas(R):
    m, B, K = R["metricas_test"], R["bandas"], R["cluster"]
    k = mejor(m)
    nivel = "moderada" if 0.3 <= m[k]["R2"] < 0.6 else "alta" if m[k]["R2"] >= 0.6 else "limitada"
    p1 = (f"Perfiles identificados: con K = {K['k']} se distinguen " + lista([p["segmento"] for p in K["perfiles"]])
          + ". Se organizan en dos ejes, edad con antigüedad y jornada habitual; "
          + f"{max(K['perfiles'], key=lambda p: p['salario_mediano'])['segmento']} tiene el mayor salario mediano y "
          + f"{min(K['perfiles'], key=lambda p: p['salario_mediano'])['segmento']} el menor.")
    p2 = (f"Capacidad de estimación del salario: con los seis predictores permitidos, {NOM[k]} alcanza en 2026T1 un MAE de "
          f"{q(m[k]['MAE'])}, un RMSE de {q(m[k]['RMSE'])} y un R² de {m[k]['R2']:.3f}, frente a un MAE de "
          f"{q(m['base']['MAE'])} y un RMSE de {q(m['base']['RMSE'])} del modelo de referencia. La capacidad predictiva es "
          f"{nivel}: el modelo reduce claramente el error respecto de la media y es estable en el tiempo, pero pierde "
          f"precisión en la cola alta, donde los salarios sobre p99 presentan un error medio de {q(B[-1][f'ME_{k}'])}.")
    return p1 + "\n\n" + p2


SECCIONES = [("Calidad y población", txt_calidad), ("Exploración salarial", txt_exploracion),
             ("Relaciones numéricas", txt_relaciones), ("Segmentación", txt_segmentacion),
             ("Modelado supervisado", txt_modelado), ("Errores", txt_errores), ("Limitaciones", txt_limitaciones),
             ("Respuesta a las preguntas centrales", txt_respuestas)]


def discusion(R):
    return [(titulo, f(R)) for titulo, f in SECCIONES]
