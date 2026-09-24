# Laboratorio 7 — Spark MLlib (CC3066 Data Science, UVG 2026)

Perfiles de trabajadores asalariados (KMeans) y predicción del salario mensual (regresión lineal y Random Forest) con la ENEIC del INE de Guatemala, usando Spark 3.5 y `pyspark.ml`.

## Estructura

```
Lab7_SparkMLlib.ipynb      Notebook con todo el procedimiento e interpretación
docker-compose.yml         Entorno Jupyter + Spark 3.5.0
data/
  raw/personas/            Excel de Personas ENEIC 2025T1–T4 y 2026T1 (no versionados)
  raw/hogares/, raw/sav/   Archivos del INE que no se usan en el análisis
  diccionarios/            Diccionarios de datos de cada archivo
  processed/               Parquet generados por el notebook (no versionados)
models/                    Pipelines guardados (mejor en validación y finales)
```

## Datos

Descargar las bases de **Personas** (.xlsx) de <https://www.ine.gob.gt/encuesta-nacional-de-empleo-e-ingresos/> y colocarlas en `data/raw/personas/` con los nombres originales del INE:

- `Personas_ENEIC_T1_2025.xlsx`
- `Personas-ENEIC-T2-2025.xlsx`
- `Base-de-datos-Personas-ENEIC-III-2025.xlsx`
- `Base-de-datos-Personas-ENEIC-IV-2025.xlsx`
- `Base-de-datos-Personas-ENEIC-I-2026.xlsx`

## Ejecución

```bash
docker compose up -d
```

Abrir <http://localhost:8888>, abrir `Lab7_SparkMLlib.ipynb` y ejecutar *Restart & Run All*. La primera ejecución convierte los Excel a Parquet (~6 min); las siguientes reutilizan esos Parquet.
