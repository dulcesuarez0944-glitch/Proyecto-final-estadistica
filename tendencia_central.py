import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
GRAFICAS    = BASE_DIR / "graficas"
OUTPUT_DIR  = BASE_DIR / "output"
GRAFICAS.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Estilo visual ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

# ── Cargar datos ───────────────────────────────────────────────────────────
print("Cargando datos...")
df = pd.read_csv(DATA_DIR / "france_main.csv")

# Variable principal: valor comercial en miles USD, sin nulos, sin ceros
datos = df[df['valor'] > 0]['valor'].dropna()
print(f"Observaciones válidas: {len(datos):,}")

# ══════════════════════════════════════════════════════════════════════════
# DATOS NO AGRUPADOS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  MEDIDAS NO AGRUPADAS")
print("="*55)

media_na    = datos.mean()
mediana_na  = datos.median()
moda_na     = datos.mode()[0]

print(f"  Media    : {media_na:>15,.2f}  miles USD")
print(f"  Mediana  : {mediana_na:>15,.2f}  miles USD")
print(f"  Moda     : {moda_na:>15,.2f}  miles USD")

# Guardar tabla no agrupada
tabla_na = pd.DataFrame({
    'Medida': ['Media', 'Mediana', 'Moda'],
    'Valor (miles USD)': [media_na, mediana_na, moda_na],
    'Interpretación': [
        'Promedio aritmético de todos los valores comerciales',
        '50% de las observaciones están por debajo de este valor',
        'Valor comercial que más se repite en la distribución'
    ]
})
tabla_na.to_csv(OUTPUT_DIR / "tendencia_no_agrupada.csv", index=False)
print("\n  ✓ Tabla guardada: tendencia_no_agrupada.csv")

# ══════════════════════════════════════════════════════════════════════════
# DATOS AGRUPADOS — Tabla de frecuencias
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  MEDIDAS AGRUPADAS")
print("="*55)

# Crear 8 intervalos de igual amplitud
n_clases = 8
minimo   = datos.min()
maximo   = datos.max()
amplitud = (maximo - minimo) / n_clases

bins = [minimo + i * amplitud for i in range(n_clases + 1)]
labels = [
    f"[{bins[i]:,.0f} – {bins[i+1]:,.0f})"
    for i in range(n_clases)
]

datos_cortados = pd.cut(datos, bins=bins, labels=labels, include_lowest=True)
tabla_frec = datos_cortados.value_counts().sort_index().reset_index()
tabla_frec.columns = ['Intervalo', 'Frecuencia']
tabla_frec['Frecuencia Relativa'] = tabla_frec['Frecuencia'] / tabla_frec['Frecuencia'].sum()
tabla_frec['Frecuencia Acumulada'] = tabla_frec['Frecuencia'].cumsum()
tabla_frec['Marca de Clase'] = [
    (bins[i] + bins[i+1]) / 2 for i in range(n_clases)
]

print("\n  Tabla de distribución de frecuencias:")
print(tabla_frec[['Intervalo','Frecuencia','Frecuencia Relativa','Marca de Clase']].to_string(index=False))

# ── Media agrupada ─────────────────────────────────────────────────────────
fi   = tabla_frec['Frecuencia'].values
xi   = tabla_frec['Marca de Clase'].values
n    = fi.sum()
media_ag = np.sum(fi * xi) / n
print(f"\n  Media agrupada   : {media_ag:>15,.2f}  miles USD")

# ── Mediana agrupada ───────────────────────────────────────────────────────
n2  = n / 2
fac = 0  # frecuencia acumulada anterior
L   = None
f_m = None
for i, row in tabla_frec.iterrows():
    if fac + row['Frecuencia'] >= n2:
        L   = bins[i]
        f_m = row['Frecuencia']
        break
    fac += row['Frecuencia']
mediana_ag = L + ((n2 - fac) / f_m) * amplitud
print(f"  Mediana agrupada : {mediana_ag:>15,.2f}  miles USD")

# ── Moda agrupada ──────────────────────────────────────────────────────────
idx_modal = tabla_frec['Frecuencia'].idxmax()
L_mo  = bins[idx_modal]
f_mo  = tabla_frec.loc[idx_modal, 'Frecuencia']
f_ant = tabla_frec.loc[idx_modal - 1, 'Frecuencia'] if idx_modal > 0 else 0
f_sig = tabla_frec.loc[idx_modal + 1, 'Frecuencia'] if idx_modal < n_clases - 1 else 0
d1    = f_mo - f_ant
d2    = f_mo - f_sig
moda_ag = L_mo + (d1 / (d1 + d2)) * amplitud if (d1 + d2) != 0 else L_mo
print(f"  Moda agrupada    : {moda_ag:>15,.2f}  miles USD")

# Guardar tabla agrupada
tabla_frec.to_csv(OUTPUT_DIR / "tabla_frecuencias.csv", index=False)

resumen_ag = pd.DataFrame({
    'Medida': ['Media agrupada', 'Mediana agrupada', 'Moda agrupada'],
    'Valor (miles USD)': [media_ag, mediana_ag, moda_ag],
})
resumen_ag.to_csv(OUTPUT_DIR / "tendencia_agrupada.csv", index=False)
print("\n  ✓ Tablas guardadas: tabla_frecuencias.csv, tendencia_agrupada.csv")

# ══════════════════════════════════════════════════════════════════════════
# GRÁFICA 1 — Histograma con media, mediana y moda (no agrupado)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(datos, bins=50, color="#4C72B0", alpha=0.75, edgecolor='white', linewidth=0.5)
ax.axvline(media_na,   color="#C44E52", linewidth=2, linestyle='-',  label=f'Media: {media_na:,.0f}')
ax.axvline(mediana_na, color="#55A868", linewidth=2, linestyle='--', label=f'Mediana: {mediana_na:,.0f}')
ax.axvline(moda_na,    color="#DD8452", linewidth=2, linestyle=':',  label=f'Moda: {moda_na:,.0f}')

ax.set_title('Distribución del Valor Comercial de Francia\ncon Medidas de Tendencia Central', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Valor Comercial (miles de USD)', fontsize=11)
ax.set_ylabel('Frecuencia', fontsize=11)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
plt.tight_layout()
plt.savefig(GRAFICAS / "01_histograma_tendencia_central.png", bbox_inches='tight')
plt.close()
print("\n  ✓ Gráfica guardada: 01_histograma_tendencia_central.png")

# ══════════════════════════════════════════════════════════════════════════
# GRÁFICA 2 — Histograma agrupado con marcas de clase
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

colores_barras = [PALETTE[0]] * n_clases
colores_barras[idx_modal] = PALETTE[3]  # resaltar clase modal

ax.bar(range(n_clases), tabla_frec['Frecuencia'],
       color=colores_barras, edgecolor='white', linewidth=0.8, alpha=0.85)

ax.axvline(idx_modal, color="#C44E52", linewidth=0, alpha=0)  # invisible, solo para leyenda
ax.set_xticks(range(n_clases))
ax.set_xticklabels(tabla_frec['Intervalo'], rotation=35, ha='right', fontsize=8)
ax.set_title('Distribución de Frecuencias Agrupadas\nValor Comercial de Francia (miles USD)', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Intervalos de Valor Comercial', fontsize=11)
ax.set_ylabel('Frecuencia', fontsize=11)

parche_modal  = mpatches.Patch(color=PALETTE[3], label='Clase modal')
parche_normal = mpatches.Patch(color=PALETTE[0], label='Otras clases')
ax.legend(handles=[parche_normal, parche_modal], fontsize=10)

plt.tight_layout()
plt.savefig(GRAFICAS / "02_histograma_agrupado.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 02_histograma_agrupado.png")

# ══════════════════════════════════════════════════════════════════════════
# GRÁFICA 3 — Comparación Media / Mediana / Moda (agrupada vs no agrupada)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

medidas  = ['Media', 'Mediana', 'Moda']
no_agrup = [media_na,   mediana_na,  moda_na]
agrup    = [media_ag,   mediana_ag,  moda_ag]

x     = np.arange(len(medidas))
ancho = 0.35

bars1 = ax.bar(x - ancho/2, no_agrup, ancho, label='No agrupada', color=PALETTE[0], alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + ancho/2, agrup,    ancho, label='Agrupada',    color=PALETTE[1], alpha=0.85, edgecolor='white')

ax.set_title('Comparación de Medidas de Tendencia Central\nAgrupadas vs No Agrupadas — Francia', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Valor (miles de USD)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(medidas, fontsize=12)
ax.legend(fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y/1e6:.2f}M'))

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
            f'{bar.get_height()/1e6:.2f}M', ha='center', va='bottom', fontsize=8, color='#333')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
            f'{bar.get_height()/1e6:.2f}M', ha='center', va='bottom', fontsize=8, color='#333')

plt.tight_layout()
plt.savefig(GRAFICAS / "03_comparacion_tendencia_central.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 03_comparacion_tendencia_central.png")

# ══════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  RESUMEN FINAL — TENDENCIA CENTRAL")
print("="*55)
print(f"\n  {'Medida':<22} {'No agrupada':>15} {'Agrupada':>15}")
print(f"  {'-'*52}")
print(f"  {'Media':<22} {media_na:>15,.2f} {media_ag:>15,.2f}")
print(f"  {'Mediana':<22} {mediana_na:>15,.2f} {mediana_ag:>15,.2f}")
print(f"  {'Moda':<22} {moda_na:>15,.2f} {moda_ag:>15,.2f}")
print(f"\n  Unidades: miles de USD")
print(f"  Observaciones usadas: {len(datos):,}")
print("\n  ✅ Fase 3 completada.")
print("  Revisa las carpetas /graficas/ y /output/")