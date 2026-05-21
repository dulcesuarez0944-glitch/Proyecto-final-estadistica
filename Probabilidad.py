"""
=====================================================================
PROYECTO FINAL - ESTADÍSTICA
Fase 5 — Introducción a la Probabilidad
País: Francia | Fuente: OCDE TEC ISIC4
=====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from itertools import combinations, permutations
from math import factorial, comb, perm
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

# ── Cargar datos ───────────────────────────────────────────────────────────
print("Cargando datos...")
df      = pd.read_csv(DATA_DIR / "france_clean.csv")
df_main = pd.read_csv(DATA_DIR / "france_main.csv")
df_usd  = pd.read_csv(DATA_DIR / "france_usd.csv")

df_valido = df_usd[df_usd['valor'] > 0].copy()
print(f"Observaciones válidas: {len(df_valido):,}")

# ══════════════════════════════════════════════════════════════════════════
# 4.1 DIAGRAMA DE VENN
# Conjunto A = Exportaciones | Conjunto B = Empresas Grandes
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  4.1 DIAGRAMA DE VENN")
print("="*60)

total   = len(df_valido)
A       = df_valido[df_valido['flujo'] == 'X']
B       = df_valido[df_valido['tamano_empresa'] == 'S_GE250']
A_y_B   = df_valido[(df_valido['flujo'] == 'X') & (df_valido['tamano_empresa'] == 'S_GE250')]
A_o_B   = df_valido[(df_valido['flujo'] == 'X') | (df_valido['tamano_empresa'] == 'S_GE250')]
solo_A  = len(A) - len(A_y_B)
solo_B  = len(B) - len(A_y_B)
intersec = len(A_y_B)
ninguno = total - len(A_o_B)

print(f"\n  Total observaciones (USD, valor > 0) : {total:,}")
print(f"  A = Exportaciones                    : {len(A):,}")
print(f"  B = Empresas Grandes (≥250 emp)      : {len(B):,}")
print(f"  A ∩ B (Exportaciones de Emp. Grande) : {intersec:,}")
print(f"  A ∪ B (Exportaciones o Emp. Grande)  : {len(A_o_B):,}")
print(f"  Solo A (Exportaciones, no Emp.Grande): {solo_A:,}")
print(f"  Solo B (Emp.Grande, no Exportaciones): {solo_B:,}")
print(f"  Ninguno                              : {ninguno:,}")

print(f"""
  INTERPRETACIÓN:
  De las {total:,} observaciones de comercio exterior de Francia,
  {len(A):,} corresponden a exportaciones (Conjunto A) y {len(B):,}
  pertenecen a empresas grandes con 250 o más empleados (Conjunto B).
  La intersección muestra que {intersec:,} observaciones son
  simultáneamente exportaciones realizadas por empresas grandes,
  lo que refleja el peso que tienen las grandes corporaciones
  francesas en el comercio internacional.
""")

# Gráfica Venn
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.set_aspect('equal')
ax.axis('off')

circulo_A = Circle((3.5, 3.5), 2.2, color='#4C72B0', alpha=0.4, label='A: Exportaciones')
circulo_B = Circle((6.5, 3.5), 2.2, color='#DD8452', alpha=0.4, label='B: Emp. Grandes')
ax.add_patch(circulo_A)
ax.add_patch(circulo_B)

ax.text(2.2, 3.5, f'Solo A\n{solo_A:,}',   ha='center', va='center', fontsize=10, fontweight='bold', color='#1a1a2e')
ax.text(7.8, 3.5, f'Solo B\n{solo_B:,}',   ha='center', va='center', fontsize=10, fontweight='bold', color='#1a1a2e')
ax.text(5.0, 3.5, f'A∩B\n{intersec:,}',    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
ax.text(0.4, 6.5, f'Ninguno: {ninguno:,}', ha='left',   va='center', fontsize=9,  color='gray')

ax.text(2.2, 6.0, 'A: Exportaciones', ha='center', fontsize=10, color='#4C72B0', fontweight='bold')
ax.text(7.8, 6.0, 'B: Emp. Grandes',  ha='center', fontsize=10, color='#DD8452', fontweight='bold')
ax.set_title('Diagrama de Venn — Comercio Exterior de Francia\nExportaciones vs Empresas Grandes (≥250 empleados)',
             fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig(GRAFICAS / "08_diagrama_venn.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 08_diagrama_venn.png")

# ══════════════════════════════════════════════════════════════════════════
# 4.2 CONTEO
# ¿De cuántas formas elegir 3 países socios de los 73 disponibles?
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  4.2 CONTEO — Principio Multiplicativo")
print("="*60)

paises      = df_valido['pais_socio'].nunique()
flujos      = df_valido['flujo'].nunique()
tamaños     = df_valido['tamano_empresa'].nunique()
años        = df_valido['año'].nunique()
conteo_total = paises * flujos * tamaños * años

print(f"\n  Países socios disponibles : {paises}")
print(f"  Tipos de flujo            : {flujos}  (X, M)")
print(f"  Tamaños de empresa        : {tamaños}")
print(f"  Años disponibles          : {años}  (2017–2024)")
print(f"\n  Total combinaciones posibles:")
print(f"  {paises} × {flujos} × {tamaños} × {años} = {conteo_total:,}")

print(f"""
  INTERPRETACIÓN:
  Aplicando el principio multiplicativo, existen {conteo_total:,}
  combinaciones posibles de análisis distintos considerando
  los {paises} países socios, {flujos} tipos de flujo comercial,
  {tamaños} categorías de tamaño de empresa y {años} años del
  período estudiado. Esto refleja la riqueza y complejidad
  del comercio exterior francés.
""")

# Gráfica conteo
fig, ax = plt.subplots(figsize=(9, 5))
categorias = ['Países\nsocios', 'Tipos de\nflujo', 'Tamaños de\nempresa', 'Años\ndisponibles']
valores    = [paises, flujos, tamaños, años]
colores    = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

bars = ax.bar(categorias, valores, color=colores, alpha=0.85, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, valores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_title('Principio de Conteo — Variables del Comercio Exterior de Francia\n'
             f'Total de combinaciones posibles: {conteo_total:,}',
             fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Cantidad de categorías', fontsize=11)
plt.tight_layout()
plt.savefig(GRAFICAS / "09_conteo.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 09_conteo.png")

# ══════════════════════════════════════════════════════════════════════════
# 4.3 COMBINACIÓN
# Seleccionar 4 tamaños de empresa de los 6 disponibles (sin orden)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  4.3 COMBINACIÓN")
print("="*60)

n_tam = 6   # tamaños disponibles (sin contar _T y _X que son agrupaciones)
r_tam = 4   # cuántos seleccionar
combinaciones_result = comb(n_tam, r_tam)

tamaños_lista = ['Micro (S0T9)', 'Pequeña (S10T49)', 'Mediana (S50T249)',
                 'Grande (S_GE250)', 'PYME (S0T249)', 'No clasificado (_X)']

print(f"\n  n = {n_tam} tamaños de empresa disponibles")
print(f"  r = {r_tam} tamaños a seleccionar para el reporte")
print(f"\n  Fórmula: C(n,r) = n! / (r! × (n-r)!)")
print(f"  C({n_tam},{r_tam}) = {n_tam}! / ({r_tam}! × {n_tam-r_tam}!)")
print(f"         = {factorial(n_tam)} / ({factorial(r_tam)} × {factorial(n_tam-r_tam)})")
print(f"         = {combinaciones_result}")

print(f"\n  Tamaños disponibles:")
for i, t in enumerate(tamaños_lista, 1):
    print(f"    {i}. {t}")

print(f"\n  Algunas combinaciones posibles:")
combs = list(combinations(tamaños_lista, r_tam))
for i, c in enumerate(combs[:5], 1):
    print(f"    {i}. {' | '.join(c)}")
print(f"    ... y {combinaciones_result - 5} más")

print(f"""
  INTERPRETACIÓN:
  Si un analista desea seleccionar {r_tam} categorías de tamaño
  de empresa de las {n_tam} disponibles en el dataset de Francia
  para elaborar un reporte comparativo, existen {combinaciones_result}
  formas posibles de hacerlo. Al tratarse de una combinación,
  el orden de selección no importa: elegir Micro, Pequeña,
  Mediana y Grande es equivalente a elegir Grande, Mediana,
  Pequeña y Micro.
""")

# Gráfica combinación
fig, ax = plt.subplots(figsize=(9, 5))
ns = list(range(1, n_tam + 1))
combs_vals = [comb(n_tam, r) for r in ns]

ax.bar(ns, combs_vals, color='#4C72B0', alpha=0.8, edgecolor='white')
ax.bar(r_tam, comb(n_tam, r_tam), color='#C44E52', alpha=0.9, edgecolor='white', label=f'C({n_tam},{r_tam}) = {combinaciones_result}')
for i, v in zip(ns, combs_vals):
    ax.text(i, v + 0.3, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_title(f'Combinaciones C({n_tam}, r) — Selección de Tamaños de Empresa\nFrancia | n = {n_tam} categorías disponibles',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('r (cantidad a seleccionar)', fontsize=11)
ax.set_ylabel('Número de combinaciones', fontsize=11)
ax.set_xticks(ns)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(GRAFICAS / "10_combinaciones.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 10_combinaciones.png")

# ══════════════════════════════════════════════════════════════════════════
# 4.4 PERMUTACIÓN
# ¿De cuántas formas ordenar los 8 años disponibles?
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  4.4 PERMUTACIÓN")
print("="*60)

n_años = 8   # años disponibles (2017-2024)
r_años = 3   # seleccionar 3 para presentar en un reporte
perm_result = perm(n_años, r_años)
perm_total  = factorial(n_años)

print(f"\n  Años disponibles: 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024")
print(f"  n = {n_años} años")
print(f"\n  Permutación total (ordenar los {n_años} años):")
print(f"  P({n_años}) = {n_años}! = {perm_total:,}")
print(f"\n  Permutación parcial (ordenar {r_años} de {n_años} años):")
print(f"  P({n_años},{r_años}) = {n_años}! / ({n_años}-{r_años})! = {perm_result:,}")

print(f"""
  INTERPRETACIÓN:
  Para presentar un análisis de la evolución del comercio
  exterior de Francia considerando los {n_años} años disponibles
  (2017–2024), existen {perm_total:,} formas distintas de
  ordenarlos en un reporte. Si solo se eligen {r_años} años
  para comparar, el número de ordenaciones posibles es de
  {perm_result:,}. A diferencia de la combinación, aquí
  el orden importa: presentar 2017→2020→2023 es diferente
  a presentar 2023→2017→2020.
""")

# Gráfica permutación
fig, ax = plt.subplots(figsize=(9, 5))
ns_p = list(range(1, n_años + 1))
perms_vals = [perm(n_años, r) for r in ns_p]

ax.plot(ns_p, perms_vals, color='#4C72B0', linewidth=2.5, marker='o', markersize=8)
ax.fill_between(ns_p, perms_vals, alpha=0.15, color='#4C72B0')
ax.scatter([r_años], [perm_result], color='#C44E52', s=120, zorder=5,
           label=f'P({n_años},{r_años}) = {perm_result:,}')

for i, v in zip(ns_p, perms_vals):
    ax.text(i, v * 1.05, f'{v:,}', ha='center', va='bottom', fontsize=8)

ax.set_title(f'Permutaciones P({n_años}, r) — Ordenamiento de Años\nComercio Exterior Francia 2017–2024',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('r (años a ordenar)', fontsize=11)
ax.set_ylabel('Número de permutaciones', fontsize=11)
ax.set_xticks(ns_p)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y):,}'))
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(GRAFICAS / "11_permutaciones.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 11_permutaciones.png")

# ══════════════════════════════════════════════════════════════════════════
# 4.5 PROBABILIDAD CONDICIONAL
# P(Exportación | Empresa Grande)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  4.5 PROBABILIDAD CONDICIONAL")
print("="*60)

total_n   = len(df_valido)
P_A       = len(df_valido[df_valido['flujo'] == 'X']) / total_n
P_B       = len(df_valido[df_valido['tamano_empresa'] == 'S_GE250']) / total_n
P_AyB     = len(df_valido[(df_valido['flujo'] == 'X') & (df_valido['tamano_empresa'] == 'S_GE250')]) / total_n
P_A_dado_B = P_AyB / P_B
P_B_dado_A = P_AyB / P_A

print(f"\n  P(A) = P(Exportación)               = {P_A:.4f}  ({P_A*100:.2f}%)")
print(f"  P(B) = P(Empresa Grande)             = {P_B:.4f}  ({P_B*100:.2f}%)")
print(f"  P(A∩B) = P(Export. y Emp. Grande)    = {P_AyB:.4f}  ({P_AyB*100:.2f}%)")
print(f"\n  P(A|B) = P(Export. | Emp. Grande)    = {P_A_dado_B:.4f}  ({P_A_dado_B*100:.2f}%)")
print(f"  P(B|A) = P(Emp. Grande | Export.)    = {P_B_dado_A:.4f}  ({P_B_dado_A*100:.2f}%)")

print(f"""
  INTERPRETACIÓN:
  La probabilidad de que una observación sea una exportación
  dado que proviene de una empresa grande es del {P_A_dado_B*100:.2f}%.
  Esto significa que dentro del subconjunto de empresas grandes
  francesas, prácticamente la mitad de sus operaciones son
  exportaciones. Por otro lado, dado que una observación es
  una exportación, la probabilidad de que corresponda a una
  empresa grande es del {P_B_dado_A*100:.2f}%.
""")

# Gráfica probabilidad condicional
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

probs    = [P_A, P_B, P_AyB, P_A_dado_B, P_B_dado_A]
etiquetas = ['P(A)\nExportación', 'P(B)\nEmp. Grande', 'P(A∩B)\nAmbas',
             'P(A|B)\nExport.|Emp.Grande', 'P(B|A)\nEmp.Grande|Export.']
colores  = ['#4C72B0', '#DD8452', '#8172B2', '#C44E52', '#55A868']

axes[0].bar(etiquetas, probs, color=colores, alpha=0.85, edgecolor='white')
for i, v in enumerate(probs):
    axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0].set_title('Probabilidades — Exportaciones\ny Empresas Grandes', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Probabilidad', fontsize=10)
axes[0].set_ylim(0, max(probs) * 1.2)

# Heatmap de probabilidades condicionales
flujos_u  = ['X', 'M']
tamaños_u = ['S0T9', 'S10T49', 'S50T249', 'S_GE250']
matriz    = np.zeros((len(tamaños_u), len(flujos_u)))

for i, tam in enumerate(tamaños_u):
    for j, flu in enumerate(flujos_u):
        subset = df_valido[df_valido['tamano_empresa'] == tam]
        if len(subset) > 0:
            matriz[i, j] = len(subset[subset['flujo'] == flu]) / len(subset)

im = axes[1].imshow(matriz, cmap='Blues', aspect='auto', vmin=0, vmax=1)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Exportaciones', 'Importaciones'])
axes[1].set_yticks(range(len(tamaños_u)))
axes[1].set_yticklabels(['Micro\n(0-9)', 'Pequeña\n(10-49)', 'Mediana\n(50-249)', 'Grande\n(≥250)'])
axes[1].set_title('P(Flujo | Tamaño de Empresa)\nHeatmap de Probabilidades Condicionales', fontsize=11, fontweight='bold')

for i in range(len(tamaños_u)):
    for j in range(len(flujos_u)):
        axes[1].text(j, i, f'{matriz[i,j]:.3f}', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if matriz[i,j] > 0.6 else 'black')

plt.colorbar(im, ax=axes[1], label='Probabilidad')
plt.tight_layout()
plt.savefig(GRAFICAS / "12_probabilidad_condicional.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 12_probabilidad_condicional.png")

# ══════════════════════════════════════════════════════════════════════════
# 4.6 TEOREMA DE BAYES
# P(Exportación | Valor Alto) usando Bayes
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  4.6 TEOREMA DE BAYES")
print("="*60)

media_val  = df_valido['valor'].mean()
alto       = df_valido[df_valido['valor'] > media_val]
export_    = df_valido[df_valido['flujo'] == 'X']
import_    = df_valido[df_valido['flujo'] == 'M']

# Probabilidades a priori
P_E  = len(export_) / total_n
P_I  = len(import_) / total_n

# Verosimilitudes
P_alto_dado_E = len(alto[alto['flujo'] == 'X']) / len(export_)
P_alto_dado_I = len(alto[alto['flujo'] == 'M']) / len(import_)

# Probabilidad total de valor alto
P_alto = (P_alto_dado_E * P_E) + (P_alto_dado_I * P_I)

# Teorema de Bayes
P_E_dado_alto = (P_alto_dado_E * P_E) / P_alto
P_I_dado_alto = (P_alto_dado_I * P_I) / P_alto

print(f"\n  Media del valor comercial: {media_val:,.2f} miles USD")
print(f"\n  PROBABILIDADES A PRIORI:")
print(f"  P(Exportación)           = {P_E:.4f}  ({P_E*100:.2f}%)")
print(f"  P(Importación)           = {P_I:.4f}  ({P_I*100:.2f}%)")
print(f"\n  VEROSIMILITUDES:")
print(f"  P(Valor Alto | Export.)  = {P_alto_dado_E:.4f}  ({P_alto_dado_E*100:.2f}%)")
print(f"  P(Valor Alto | Import.)  = {P_alto_dado_I:.4f}  ({P_alto_dado_I*100:.2f}%)")
print(f"\n  P(Valor Alto) total      = {P_alto:.4f}  ({P_alto*100:.2f}%)")
print(f"\n  RESULTADOS TEOREMA DE BAYES:")
print(f"  P(Export. | Valor Alto)  = {P_E_dado_alto:.4f}  ({P_E_dado_alto*100:.2f}%)")
print(f"  P(Import. | Valor Alto)  = {P_I_dado_alto:.4f}  ({P_I_dado_alto*100:.2f}%)")

print(f"""
  INTERPRETACIÓN:
  Aplicando el Teorema de Bayes, si observamos un valor
  comercial superior a la media ({media_val:,.0f} miles USD),
  la probabilidad de que ese intercambio sea una exportación
  es del {P_E_dado_alto*100:.2f}%, mientras que la probabilidad
  de que sea una importación es del {P_I_dado_alto*100:.2f}%.
  Esto sugiere que los valores comerciales más altos de
  Francia tienden a asociarse {'más con exportaciones' if P_E_dado_alto > P_I_dado_alto else 'más con importaciones'},
  lo que refleja la fortaleza exportadora del sector
  empresarial francés en los intercambios de mayor volumen.
""")

# Gráfica Bayes
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Priori vs Posteriori
categorias_b = ['Exportación', 'Importación']
priori       = [P_E, P_I]
posteriori   = [P_E_dado_alto, P_I_dado_alto]

x_b    = np.arange(len(categorias_b))
ancho  = 0.35
axes[0].bar(x_b - ancho/2, priori,     ancho, label='A priori',    color='#4C72B0', alpha=0.8, edgecolor='white')
axes[0].bar(x_b + ancho/2, posteriori, ancho, label='A posteriori',color='#C44E52', alpha=0.8, edgecolor='white')

for i, (p, po) in enumerate(zip(priori, posteriori)):
    axes[0].text(i - ancho/2, p  + 0.005, f'{p:.3f}',  ha='center', fontsize=10, fontweight='bold')
    axes[0].text(i + ancho/2, po + 0.005, f'{po:.3f}', ha='center', fontsize=10, fontweight='bold')

axes[0].set_title('Teorema de Bayes\nPriori vs Posteriori (dado Valor Alto)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Probabilidad', fontsize=10)
axes[0].set_xticks(x_b)
axes[0].set_xticklabels(categorias_b)
axes[0].legend(fontsize=10)
axes[0].set_ylim(0, 0.8)

# Diagrama de flujo Bayes
axes[1].axis('off')
texto_bayes = (
    "TEOREMA DE BAYES\n\n"
    "P(E|Alto) = P(Alto|E) × P(E)\n"
    "            ─────────────────\n"
    "                 P(Alto)\n\n"
    f"= {P_alto_dado_E:.4f} × {P_E:.4f}\n"
    f"  ─────────────────────\n"
    f"       {P_alto:.4f}\n\n"
    f"= {P_E_dado_alto:.4f} ({P_E_dado_alto*100:.2f}%)\n\n"
    "Donde:\n"
    f"  P(E)        = {P_E:.4f}  (exportaciones)\n"
    f"  P(Alto|E)   = {P_alto_dado_E:.4f}  (valor alto dado export.)\n"
    f"  P(Alto)     = {P_alto:.4f}  (prob. total valor alto)"
)
axes[1].text(0.05, 0.95, texto_bayes, transform=axes[1].transAxes,
             fontsize=10, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.8))
axes[1].set_title('Desarrollo del Cálculo', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(GRAFICAS / "13_teorema_bayes.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 13_teorema_bayes.png")

# ══════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  RESUMEN — PROBABILIDAD")
print("="*60)
print(f"\n  P(Exportación)                = {P_A:.4f}")
print(f"  P(Empresa Grande)             = {P_B:.4f}")
print(f"  P(Export. ∩ Emp. Grande)      = {P_AyB:.4f}")
print(f"  P(Export. | Emp. Grande)      = {P_A_dado_B:.4f}")
print(f"  P(Export. | Valor Alto)       = {P_E_dado_alto:.4f}  ← Bayes")
print(f"\n  Combinaciones C(6,4)          = {combinaciones_result}")
print(f"  Permutaciones P(8,3)          = {perm_result:,}")
print(f"  Conteo total posible          = {conteo_total:,}")
print("\n  ✅ Fase 5 completada.")
print("  Revisa las carpetas /graficas/ y /output/")