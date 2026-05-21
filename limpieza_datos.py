"""
=====================================================================
PROYECTO FINAL - ESTADÍSTICA
Descarga desde API OCDE + Limpieza de datos: Francia
=====================================================================
Este script se conecta a la API oficial de OCDE Data Explorer,
descarga los datos de comercio exterior de Francia y los limpia
para su análisis estadístico.
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO

# ── Rutas relativas ────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("  PROYECTO FINAL — ESTADÍSTICA")
print("  Comercio Exterior de Francia | OCDE TEC ISIC4")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════
# PASO 1 — CONEXIÓN A LA API DE OCDE DATA EXPLORER
# ══════════════════════════════════════════════════════════════════
print("\n[PASO 1] Conectando a la API de OCDE Data Explorer...")
print("  Fuente: https://data-explorer.oecd.org/")
print("  Dataset: Trade by Enterprise Characteristics (TEC ISIC4)")
print("  País: Francia (FRA) | Período: 2017-2024")

API_URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.SDD.TPS,DSD_TEC_ISIC4,1.1/"
    "FRA.........?"
    "startPeriod=2017&endPeriod=2024"
    "&format=csvfilewithlabels"
    "&dimensionAtObservation=AllDimensions"
)

print(f"\n  Endpoint: {API_URL[:55]}...")

try:
    print("  Enviando solicitud a la API...")
    response = requests.get(API_URL, timeout=60)
    response.raise_for_status()
    raw = pd.read_csv(StringIO(response.text))
    print(f"  ✓ Datos descargados exitosamente desde la API")
    print(f"  ✓ Registros recibidos: {len(raw):,} filas")
    raw.to_csv(DATA_DIR / "france_api_raw.csv", index=False)
    print(f"  ✓ Respuesta guardada en: data/france_api_raw.csv")
    fuente = "API OCDE Data Explorer"

except Exception as e:
    print(f"  ! API no disponible ({type(e).__name__})")
    print("  ! Usando archivo CSV local como respaldo...")
    local = DATA_DIR / "OECD_SDD_TPS_DSD_TEC_ISIC4_DF_TEC10_1_1_all.csv"
    if not local.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo local en:\n{local}\n"
            "Coloca el CSV original en la carpeta /data/"
        )
    raw = pd.read_csv(local)
    raw = raw[[c for c in raw.columns if not c.startswith('Unnamed')]]
    raw = raw[raw['REF_AREA'] == 'FRA']
    print(f"  ✓ Archivo local cargado: {len(raw):,} filas")
    fuente = "CSV local (OCDE)"

# ══════════════════════════════════════════════════════════════════
# PASO 2 — FILTRAR FRANCIA
# ══════════════════════════════════════════════════════════════════
print("\n[PASO 2] Filtrando registros de Francia...")

named_cols = [c for c in raw.columns if not c.startswith('Unnamed')]
france = raw[named_cols].copy()

if 'REF_AREA' in france.columns:
    france = france[france['REF_AREA'] == 'FRA'].copy()

print(f"  ✓ Registros de Francia: {len(france):,} filas")

# ══════════════════════════════════════════════════════════════════
# PASO 3 — LIMPIEZA DE DATOS
# ══════════════════════════════════════════════════════════════════
print("\n[PASO 3] Limpiando datos...")

# Eliminar columnas sin variabilidad o metadatos técnicos
drop_cols = [
    'STRUCTURE', 'STRUCTURE_ID', 'STRUCTURE_NAME', 'ACTION',
    'FREQ', 'TABLE', 'REF_AREA',
    'ACTIVITY', 'TOP_ENT', 'NBPARTNER',
    'CPC', 'TTRADER', 'TOWNERSHIP', 'EXPINT',
    'UNIT_MULT', 'DECIMALS',
]
france.drop(columns=[c for c in drop_cols if c in france.columns], inplace=True)
print(f"  ✓ Columnas eliminadas: {len(drop_cols)} columnas sin variabilidad")

# Renombrar columnas a español
france.rename(columns={
    'PARTNER_COUNTRY': 'pais_socio',
    'MEASURE':         'tipo_medida',
    'SIZE_CLASS':      'tamano_empresa',
    'FLOW':            'flujo',
    'UNIT_MEASURE':    'unidad',
    'TIME_PERIOD':     'año',
    'OBS_VALUE':       'valor',
    'OBS_STATUS':      'calidad_dato',
}, inplace=True)
print("  ✓ Columnas renombradas al español")

# Agregar etiquetas descriptivas
france['flujo_etiq'] = france['flujo'].map({
    'X': 'Exportaciones', 'M': 'Importaciones'
})
france['tamano_etiq'] = france['tamano_empresa'].map({
    '_T': 'Total', '_X': 'No clasificado',
    'S0T9':    'Micro (0-9 emp)',
    'S10T49':  'Pequeña (10-49 emp)',
    'S50T249': 'Mediana (50-249 emp)',
    'S_GE250': 'Grande (≥250 emp)',
    'S0T249':  'PYME (0-249 emp)',
})
france['calidad_etiq'] = france['calidad_dato'].map({
    'A': 'Normal', 'Q': 'Provisional', 'U': 'No disponible'
})
france['tipo_medida_etiq'] = france['tipo_medida'].map({
    'TUTT_TEC': 'Valor comercial',
    'ENTR_TEC': 'N° empresas',
})
print("  ✓ Etiquetas descriptivas agregadas")

# Convertir tipos de datos
france['año']   = pd.to_numeric(france['año'],   errors='coerce').astype('Int64')
france['valor'] = pd.to_numeric(france['valor'], errors='coerce')
print("  ✓ Tipos de datos corregidos")

# ══════════════════════════════════════════════════════════════════
# PASO 4 — CREAR SUBSETS TEMÁTICOS
# ══════════════════════════════════════════════════════════════════
print("\n[PASO 4] Creando subsets para análisis...")

df_usd  = france[(france['unidad'] == 'USD') & france['valor'].notna()].copy()
df_ent  = france[(france['unidad'] == 'ENT') & france['valor'].notna()].copy()
df_main = df_usd[df_usd['tamano_empresa'] == '_T'].copy()

france.to_csv(DATA_DIR  / "france_clean.csv", index=False)
df_usd.to_csv(DATA_DIR  / "france_usd.csv",   index=False)
df_ent.to_csv(DATA_DIR  / "france_ent.csv",   index=False)
df_main.to_csv(DATA_DIR / "france_main.csv",  index=False)

print(f"  ✓ france_clean.csv  → {len(france):,} filas (dataset completo)")
print(f"  ✓ france_usd.csv    → {len(df_usd):,} filas (valor en miles USD)")
print(f"  ✓ france_ent.csv    → {len(df_ent):,} filas (número de empresas)")
print(f"  ✓ france_main.csv   → {len(df_main):,} filas (análisis principal)")

# ══════════════════════════════════════════════════════════════════
# PASO 5 — REPORTE DE CALIDAD
# ══════════════════════════════════════════════════════════════════
print("\n[PASO 5] Reporte de calidad de datos...")

total  = len(france)
nulos  = france['valor'].isna().sum()
normal = (france['calidad_dato'] == 'A').sum()
prov   = (france['calidad_dato'] == 'Q').sum()
nodis  = (france['calidad_dato'] == 'U').sum()

print(f"\n  Total registros Francia   : {total:,}")
print(f"  Valores nulos             : {nulos:,} ({nulos/total*100:.1f}%)")
print(f"  Calidad Normal (A)        : {normal:,} ({normal/total*100:.1f}%)")
print(f"  Calidad Provisional (Q)   : {prov:,}  ({prov/total*100:.1f}%)")
print(f"  No disponibles (U)        : {nodis:,}  ({nodis/total*100:.1f}%)")
print(f"  Período cubierto          : {france['año'].min()} – {france['año'].max()}")
print(f"  Países socios             : {france['pais_socio'].nunique()} países")
print(f"  Fuente de datos           : {fuente}")

print("\n" + "=" * 60)
print("  ✅ Datos descargados y limpios correctamente.")
print("  Ahora ejecuta los scripts de análisis estadístico.")
print("=" * 60)