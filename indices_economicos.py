"""
=====================================================================
PROYECTO FINAL - ESTADÍSTICA
Fase 8 — Índices Económicos: Paasche, Laspeyres y Fisher
País: Francia | Fuente: OCDE TEC ISIC4
=====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
GRAFICAS   = BASE_DIR / "graficas"
OUTPUT_DIR = BASE_DIR / "output"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

# ── Cargar datos ───────────────────────────────────────────────────────────
print("Cargando datos...")
df_usd    = pd.read_csv(DATA_DIR / "france_usd.csv")
df_valido = df_usd[df_usd['valor'] > 0].copy()

print(f"Observaciones válidas: {len(df_valido):,}")
print(f"Años disponibles: {sorted(df_valido['año'].unique())}")

# ── Preparar datos por país socio y año ────────────────────────────────────
# Precio (p) = valor promedio por país socio y año
# Cantidad (q) = número de observaciones por país socio y año
df_grouped = df_valido.groupby(['año', 'pais_socio']).agg(
    precio=('valor', 'mean'),
    cantidad=('valor', 'count')
).reset_index()

AÑO_BASE  = 2017
AÑO_COMP  = 2022

base = df_grouped[df_grouped['año'] == AÑO_BASE].copy()
comp = df_grouped[df_grouped['año'] == AÑO_COMP].copy()

# Países comunes en ambos años
paises_comunes = set(base['pais_socio']) & set(comp['pais_socio'])
base = base[base['pais_socio'].isin(paises_comunes)].set_index('pais_socio')
comp = comp[comp['pais_socio'].isin(paises_comunes)].set_index('pais_socio')

print(f"\nAño base      : {AÑO_BASE}")
print(f"Año comparado : {AÑO_COMP}")
print(f"Países socios comunes: {len(paises_comunes)}")

p0 = base['precio'].values
q0 = base['cantidad'].values
p1 = comp['precio'].values
q1 = comp['cantidad'].values

# ══════════════════════════════════════════════════════════════════════════
# 7.1 ÍNDICE LASPEYRES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  7.1 ÍNDICE LASPEYRES")
print("="*60)

numerador_L   = np.sum(p1 * q0)
denominador_L = np.sum(p0 * q0)
IL            = (numerador_L / denominador_L) * 100

print(f"""
  PROBLEMA:
  Usando {AÑO_BASE} como año base, ¿cómo cambió el valor
  del comercio exterior de Francia en {AÑO_COMP}?
  El Índice Laspeyres usa las CANTIDADES del año base.

  FÓRMULA:
  IL = (Σ p1 × q0) / (Σ p0 × q0) × 100

  CÁLCULO:
  Σ(p1 × q0) = {numerador_L:>20,.2f}  (precios {AÑO_COMP} × cantidades {AÑO_BASE})
  Σ(p0 × q0) = {denominador_L:>20,.2f}  (precios {AÑO_BASE} × cantidades {AÑO_BASE})

  IL = ({numerador_L:,.2f} / {denominador_L:,.2f}) × 100
  IL = {IL:.4f}

  INTERPRETACIÓN:
  El Índice Laspeyres de {IL:.2f} indica que el valor
  del comercio exterior de Francia {'aumentó' if IL > 100 else 'disminuyó'}
  un {abs(IL - 100):.2f}% entre {AÑO_BASE} y {AÑO_COMP},
  manteniendo constantes las cantidades del año base {AÑO_BASE}.
  {'Esto refleja un crecimiento significativo en los valores' if IL > 100 else 'Esto refleja una contracción en los valores'}
  comerciales por país socio durante ese período,
  influenciado por factores como la inflación, cambios
  en la demanda internacional y el impacto económico
  post-pandemia en el comercio francés.
""")

# ══════════════════════════════════════════════════════════════════════════
# 7.2 ÍNDICE PAASCHE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  7.2 ÍNDICE PAASCHE")
print("="*60)

numerador_P   = np.sum(p1 * q1)
denominador_P = np.sum(p0 * q1)
IP            = (numerador_P / denominador_P) * 100

print(f"""
  PROBLEMA:
  Con los mismos datos pero usando las CANTIDADES del año
  actual ({AÑO_COMP}) como ponderadores, ¿cuánto cambió
  el comercio exterior francés respecto a {AÑO_BASE}?

  FÓRMULA:
  IP = (Σ p1 × q1) / (Σ p0 × q1) × 100

  CÁLCULO:
  Σ(p1 × q1) = {numerador_P:>20,.2f}  (precios {AÑO_COMP} × cantidades {AÑO_COMP})
  Σ(p0 × q1) = {denominador_P:>20,.2f}  (precios {AÑO_BASE} × cantidades {AÑO_COMP})

  IP = ({numerador_P:,.2f} / {denominador_P:,.2f}) × 100
  IP = {IP:.4f}

  INTERPRETACIÓN:
  El Índice Paasche de {IP:.2f} indica que usando las
  cantidades actuales de {AÑO_COMP} como ponderadores,
  el comercio exterior de Francia {'aumentó' if IP > 100 else 'disminuyó'}
  un {abs(IP - 100):.2f}% respecto a {AÑO_BASE}.
  A diferencia del Índice Laspeyres que usa cantidades
  fijas del año base, el Paasche refleja mejor la
  estructura actual del comercio francés, tomando en
  cuenta los cambios en los patrones de intercambio
  con cada país socio.
""")

# ══════════════════════════════════════════════════════════════════════════
# 7.3 ÍNDICE FISHER
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  7.3 ÍNDICE FISHER")
print("="*60)

IF = (IL * IP) ** 0.5

print(f"""
  PROBLEMA:
  El Índice Fisher combina Laspeyres y Paasche mediante
  la media geométrica para obtener un índice más equilibrado.

  FÓRMULA:
  IF = √(IL × IP)

  CÁLCULO:
  IL = {IL:.4f}
  IP = {IP:.4f}

  IF = √({IL:.4f} × {IP:.4f})
  IF = √({IL * IP:.4f})
  IF = {IF:.4f}

  COMPARACIÓN DE ÍNDICES:
  {'Índice':<25} {'Valor':>10} {'Cambio %':>12}
  {'-'*47}
  {'Laspeyres (IL)':<25} {IL:>10.4f} {IL-100:>+11.2f}%
  {'Paasche (IP)':<25} {IP:>10.4f} {IP-100:>+11.2f}%
  {'Fisher (IF)':<25} {IF:>10.4f} {IF-100:>+11.2f}%

  INTERPRETACIÓN:
  El Índice Fisher de {IF:.2f} representa la medida más
  equilibrada del cambio en el comercio exterior de Francia
  entre {AÑO_BASE} y {AÑO_COMP}. Al ser la media geométrica
  de Laspeyres ({IL:.2f}) y Paasche ({IP:.2f}), corrige los
  sesgos de cada índice individual.
  {'El Fisher supera 100, confirmando un crecimiento' if IF > 100 else 'El Fisher está por debajo de 100, confirmando una contracción'}
  del {abs(IF-100):.2f}% en el valor del comercio exterior
  francés durante ese período. Este resultado refleja
  el dinamismo de Francia como potencia comercial
  europea, con variaciones en sus intercambios con
  los {len(paises_comunes)} países socios analizados.
""")

# ── Guardar tabla resumen ──────────────────────────────────────────────────
tabla_indices = pd.DataFrame({
    'Índice': ['Laspeyres', 'Paasche', 'Fisher'],
    'Valor': [IL, IP, IF],
    'Cambio (%)': [IL-100, IP-100, IF-100],
    'Interpretación': [
        f'Cambio usando cantidades de {AÑO_BASE}',
        f'Cambio usando cantidades de {AÑO_COMP}',
        'Media geométrica de Laspeyres y Paasche'
    ]
})
tabla_indices.to_csv(OUTPUT_DIR / "indices_economicos.csv", index=False)
print("  ✓ Tabla guardada: indices_economicos.csv")

# ── Evolución anual de índices ─────────────────────────────────────────────
años = sorted(df_valido['año'].unique())
IL_serie = []
IP_serie = []
IF_serie = []

for año in años:
    comp_año = df_grouped[df_grouped['año'] == año].copy()
    paises_c = set(base.index) & set(comp_año['pais_socio'])
    if len(paises_c) < 5:
        IL_serie.append(np.nan)
        IP_serie.append(np.nan)
        IF_serie.append(np.nan)
        continue
    b = base.loc[list(paises_c)]
    c = comp_año[comp_año['pais_socio'].isin(paises_c)].set_index('pais_socio').loc[list(paises_c)]
    p0_ = b['precio'].values
    q0_ = b['cantidad'].values
    p1_ = c['precio'].values
    q1_ = c['cantidad'].values
    il  = np.sum(p1_*q0_) / np.sum(p0_*q0_) * 100
    ip  = np.sum(p1_*q1_) / np.sum(p0_*q1_) * 100
    if_ = (il * ip) ** 0.5
    IL_serie.append(il)
    IP_serie.append(ip)
    IF_serie.append(if_)

tabla_serie = pd.DataFrame({
    'Año': años,
    'Laspeyres': IL_serie,
    'Paasche':   IP_serie,
    'Fisher':    IF_serie,
})
tabla_serie.to_csv(OUTPUT_DIR / "indices_serie_temporal.csv", index=False)
print("  ✓ Tabla guardada: indices_serie_temporal.csv")

# ══════════════════════════════════════════════════════════════════════════
# GRÁFICAS
# ══════════════════════════════════════════════════════════════════════════

# Gráfica 1 — Comparación de los 3 índices
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

indices  = ['Laspeyres', 'Paasche', 'Fisher']
valores  = [IL, IP, IF]
colores  = ['#4C72B0', '#DD8452', '#55A868']

bars = axes[0].bar(indices, valores, color=colores, alpha=0.85,
                   edgecolor='white', linewidth=0.8, width=0.5)
axes[0].axhline(100, color='#C44E52', linewidth=2, linestyle='--',
                label='Base = 100 (año 2017)')
for bar, val in zip(bars, valores):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f'{val:.2f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

axes[0].set_title(f'Índices Económicos — Francia\nAño Base {AÑO_BASE} vs {AÑO_COMP}',
                  fontsize=11, fontweight='bold')
axes[0].set_ylabel('Valor del Índice', fontsize=10)
axes[0].legend(fontsize=9)
axes[0].set_ylim(min(valores)*0.9, max(valores)*1.05)

# Gráfica 2 — Evolución temporal
axes[1].plot(años, IL_serie, color='#4C72B0', linewidth=2.5,
             marker='o', markersize=7, label='Laspeyres')
axes[1].plot(años, IP_serie, color='#DD8452', linewidth=2.5,
             marker='s', markersize=7, label='Paasche')
axes[1].plot(años, IF_serie, color='#55A868', linewidth=2.5,
             marker='^', markersize=7, label='Fisher')
axes[1].axhline(100, color='#C44E52', linewidth=1.5,
                linestyle='--', label=f'Base = 100 ({AÑO_BASE})')
axes[1].fill_between(años, IL_serie, IF_serie, alpha=0.08, color='#4C72B0')

axes[1].set_title(f'Evolución de Índices Económicos\nFrancia {años[0]}–{años[-1]}',
                  fontsize=11, fontweight='bold')
axes[1].set_xlabel('Año', fontsize=10)
axes[1].set_ylabel('Valor del Índice', fontsize=10)
axes[1].set_xticks(años)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(GRAFICAS / "20_indices_economicos.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 20_indices_economicos.png")

# Gráfica 2 — Top 10 países con mayor cambio de precio
cambio_pais = pd.DataFrame({
    'pais_socio': list(paises_comunes),
    'precio_base': [base.loc[p, 'precio'] for p in paises_comunes],
    'precio_comp': [comp.loc[p, 'precio'] for p in paises_comunes],
})
cambio_pais['cambio_%'] = ((cambio_pais['precio_comp'] -
                             cambio_pais['precio_base']) /
                             cambio_pais['precio_base']) * 100
top10 = cambio_pais.nlargest(10, 'cambio_%')
bot10 = cambio_pais.nsmallest(5, 'cambio_%')
top_combined = pd.concat([top10, bot10]).drop_duplicates()

fig, ax = plt.subplots(figsize=(11, 6))
colores_cambio = ['#55A868' if v >= 0 else '#C44E52'
                  for v in top_combined['cambio_%']]
bars2 = ax.barh(top_combined['pais_socio'], top_combined['cambio_%'],
                color=colores_cambio, alpha=0.85, edgecolor='white')
ax.axvline(0, color='black', linewidth=1)
for bar, val in zip(bars2, top_combined['cambio_%']):
    ax.text(val + (2 if val >= 0 else -2),
            bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=8,
            ha='left' if val >= 0 else 'right')

ax.set_title(f'Cambio en Valor Comercial por País Socio\n'
             f'Francia {AÑO_BASE} → {AÑO_COMP} (Top y Bottom países)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Cambio porcentual en valor comercial (%)', fontsize=10)

parches2 = [
    mpatches.Patch(color='#55A868', alpha=0.85, label='Incremento'),
    mpatches.Patch(color='#C44E52', alpha=0.85, label='Decremento'),
]
ax.legend(handles=parches2, fontsize=9)
plt.tight_layout()
plt.savefig(GRAFICAS / "21_cambio_por_pais.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 21_cambio_por_pais.png")

# ══════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  RESUMEN — ÍNDICES ECONÓMICOS")
print("="*60)
print(f"\n  Año base      : {AÑO_BASE}")
print(f"  Año comparado : {AÑO_COMP}")
print(f"  Países socios : {len(paises_comunes)}")
print(f"\n  {'Índice':<20} {'Valor':>10} {'Cambio':>10}")
print(f"  {'-'*40}")
print(f"  {'Laspeyres':<20} {IL:>10.4f} {IL-100:>+9.2f}%")
print(f"  {'Paasche':<20} {IP:>10.4f} {IP-100:>+9.2f}%")
print(f"  {'Fisher':<20} {IF:>10.4f} {IF-100:>+9.2f}%")
print(f"\n  ✅ Fase 8 completada.")
print("  Revisa las carpetas /graficas/ y /output/")