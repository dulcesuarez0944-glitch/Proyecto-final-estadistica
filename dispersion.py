import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
GRAFICAS   = BASE_DIR / "graficas"
OUTPUT_DIR = BASE_DIR / "output"

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
df   = pd.read_csv(DATA_DIR / "france_main.csv")
datos = df[df['valor'] > 0]['valor'].dropna()
print(f"Observaciones válidas: {len(datos):,}")

# ══════════════════════════════════════════════════════════════════════════
# MEDIDAS DE DISPERSIÓN
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  MEDIDAS DE DISPERSIÓN")
print("="*55)

# 3.1 Rango
rango = datos.max() - datos.min()

# 3.2 Rango Intercuartílico
Q1  = datos.quantile(0.25)
Q3  = datos.quantile(0.75)
RIC = Q3 - Q1

# 3.3 Varianza
varianza = datos.var(ddof=1)

# 3.4 Desviación estándar
desv_std = datos.std(ddof=1)

# 3.5 Desviación media
desv_media = (datos - datos.mean()).abs().mean()

print(f"\n  Mínimo             : {datos.min():>20,.2f}  miles USD")
print(f"  Máximo             : {datos.max():>20,.2f}  miles USD")
print(f"  Rango              : {rango:>20,.2f}  miles USD")
print(f"  Q1 (25%)           : {Q1:>20,.2f}  miles USD")
print(f"  Q3 (75%)           : {Q3:>20,.2f}  miles USD")
print(f"  Rango Intercuart.  : {RIC:>20,.2f}  miles USD")
print(f"  Varianza           : {varianza:>20,.2f}")
print(f"  Desviación Estánd. : {desv_std:>20,.2f}  miles USD")
print(f"  Desviación Media   : {desv_media:>20,.2f}  miles USD")

# Coeficiente de variación
cv = (desv_std / datos.mean()) * 100
print(f"\n  Coef. de Variación : {cv:>19.2f}%")

# Guardar tabla
tabla_disp = pd.DataFrame({
    'Medida': [
        'Mínimo', 'Máximo', 'Rango',
        'Q1 (25%)', 'Q3 (75%)', 'Rango Intercuartílico',
        'Varianza', 'Desviación Estándar', 'Desviación Media',
        'Coeficiente de Variación (%)'
    ],
    'Valor': [
        datos.min(), datos.max(), rango,
        Q1, Q3, RIC,
        varianza, desv_std, desv_media, cv
    ],
    'Interpretación': [
        'Valor comercial más bajo registrado',
        'Valor comercial más alto registrado',
        'Amplitud total de la distribución',
        '25% de los valores están por debajo',
        '75% de los valores están por debajo',
        'Dispersión del 50% central de los datos',
        'Promedio de las desviaciones al cuadrado',
        'Dispersión promedio respecto a la media',
        'Promedio de desviaciones absolutas',
        'Variabilidad relativa respecto a la media'
    ]
})
tabla_disp.to_csv(OUTPUT_DIR / "dispersion.csv", index=False)
print("\n  ✓ Tabla guardada: dispersion.csv")

# ══════════════════════════════════════════════════════════════════════════
# GRÁFICA 1 — Boxplot general
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

bp = ax.boxplot(datos, vert=False, patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", alpha=0.6),
                medianprops=dict(color="#C44E52", linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker='o', color='#DD8452', alpha=0.4, markersize=4))

ax.axvline(datos.mean(), color="#55A868", linewidth=2, linestyle='--', label=f'Media: {datos.mean():,.0f}')
ax.set_title('Boxplot del Valor Comercial de Francia\nDistribución y Valores Atípicos', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Valor Comercial (miles de USD)', fontsize=11)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
ax.legend(fontsize=10)
ax.set_yticks([])
plt.tight_layout()
plt.savefig(GRAFICAS / "04_boxplot_general.png", bbox_inches='tight')
plt.close()
print("\n  ✓ Gráfica guardada: 04_boxplot_general.png")

# ══════════════════════════════════════════════════════════════════════════
# GRÁFICA 2 — Boxplot por flujo (Exportaciones vs Importaciones)
# ══════════════════════════════════════════════════════════════════════════
df_valido = df[df['valor'] > 0].copy()

fig, ax = plt.subplots(figsize=(9, 5))

grupos = [
    df_valido[df_valido['flujo'] == 'X']['valor'].dropna(),
    df_valido[df_valido['flujo'] == 'M']['valor'].dropna(),
]
etiquetas = ['Exportaciones', 'Importaciones']

bp2 = ax.boxplot(grupos, vert=True, patch_artist=True,
                 labels=etiquetas,
                 boxprops=dict(alpha=0.7),
                 medianprops=dict(linewidth=2),
                 flierprops=dict(marker='o', alpha=0.3, markersize=4))

colores = ["#4C72B0", "#DD8452"]
for patch, color in zip(bp2['boxes'], colores):
    patch.set_facecolor(color)

ax.set_title('Dispersión del Valor Comercial por Tipo de Flujo\nFrancia — Exportaciones vs Importaciones', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Valor Comercial (miles de USD)', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y/1e6:.0f}M'))
plt.tight_layout()
plt.savefig(GRAFICAS / "05_boxplot_flujo.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 05_boxplot_flujo.png")

# ══════════════════════════════════════════════════════════════════════════
# GRÁFICA 3 — Histograma con zonas de dispersión
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(datos, bins=60, color="#4C72B0", alpha=0.65, edgecolor='white', linewidth=0.4)

media   = datos.mean()
ax.axvline(media,              color="#C44E52", linewidth=2,   linestyle='-',  label=f'Media: {media:,.0f}')
ax.axvline(media - desv_std,   color="#55A868", linewidth=1.5, linestyle='--', label=f'Media ± 1σ')
ax.axvline(media + desv_std,   color="#55A868", linewidth=1.5, linestyle='--')
ax.axvline(media - 2*desv_std, color="#DD8452", linewidth=1.5, linestyle=':',  label=f'Media ± 2σ')
ax.axvline(media + 2*desv_std, color="#DD8452", linewidth=1.5, linestyle=':')

ax.axvspan(media - desv_std, media + desv_std,   alpha=0.08, color="#55A868")
ax.axvspan(media - 2*desv_std, media + 2*desv_std, alpha=0.05, color="#DD8452")

ax.set_title('Histograma con Zonas de Desviación Estándar\nValor Comercial de Francia', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Valor Comercial (miles de USD)', fontsize=11)
ax.set_ylabel('Frecuencia', fontsize=11)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(GRAFICAS / "06_histograma_dispersion.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 06_histograma_dispersion.png")

# ══════════════════════════════════════════════════════════════════════════
# GRÁFICA 4 — Dispersión por año
# ══════════════════════════════════════════════════════════════════════════
df_año = df_valido.groupby('año')['valor'].agg(['mean','std','median']).reset_index()

fig, ax = plt.subplots(figsize=(10, 5))

ax.fill_between(df_año['año'],
                df_año['mean'] - df_año['std'],
                df_año['mean'] + df_año['std'],
                alpha=0.2, color="#4C72B0", label='± 1 Desv. Estándar')
ax.plot(df_año['año'], df_año['mean'],   color="#4C72B0", linewidth=2.5, marker='o', label='Media')
ax.plot(df_año['año'], df_año['median'], color="#C44E52", linewidth=2,   marker='s', linestyle='--', label='Mediana')

ax.set_title('Evolución de la Dispersión del Valor Comercial\nFrancia 2017–2024', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Año', fontsize=11)
ax.set_ylabel('Valor Comercial (miles de USD)', fontsize=11)
ax.set_xticks(df_año['año'])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y/1e6:.0f}M'))
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(GRAFICAS / "07_dispersion_temporal.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 07_dispersion_temporal.png")

# ══════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  RESUMEN FINAL — DISPERSIÓN")
print("="*55)
print(f"\n  Rango              : {rango:>20,.2f}  miles USD")
print(f"  Rango Intercuart.  : {RIC:>20,.2f}  miles USD")
print(f"  Desviación Estánd. : {desv_std:>20,.2f}  miles USD")
print(f"  Desviación Media   : {desv_media:>20,.2f}  miles USD")
print(f"  Varianza           : {varianza:>20,.2f}")
print(f"  Coef. Variación    : {cv:>19.2f}%")
print(f"\n  ✅ Fase 4 completada.")
print("  Revisa las carpetas /graficas/ y /output/")