"""
=====================================================================
PROYECTO FINAL - ESTADÍSTICA
Fase 6 — Distribuciones de Probabilidad con Variable Aleatoria Discreta
País: Francia | Fuente: OCDE TEC ISIC4
=====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import binom, poisson, hypergeom
from math import comb, factorial, exp
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

total    = len(df_valido)
export_n = len(df_valido[df_valido['flujo'] == 'X'])
p_export = export_n / total
print(f"Observaciones válidas : {total:,}")
print(f"Exportaciones         : {export_n:,}")
print(f"P(Exportación)        : {p_export:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 5.1 DISTRIBUCIÓN BINOMIAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  5.1 DISTRIBUCIÓN BINOMIAL")
print("="*60)

n        = 10
p        = p_export
k_exacto = 6

print(f"""
  PROBLEMA:
  En Francia, la probabilidad de que una observación de comercio
  exterior sea una exportación es p = {p:.4f} ({p*100:.2f}%).
  Si se seleccionan aleatoriamente n = {n} observaciones,
  ¿cuál es la probabilidad de que exactamente k = {k_exacto}
  sean exportaciones? ¿Y de que al menos 7 sean exportaciones?

  PARÁMETROS:
  n = {n}      (observaciones seleccionadas)
  p = {p:.4f}  (probabilidad de exportación)
  q = {1-p:.4f}  (probabilidad de no exportación)

  FÓRMULA:
  P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
""")

P_exacto     = binom.pmf(k_exacto, n, p)
P_al_menos_7 = sum(binom.pmf(k, n, p) for k in range(7, n+1))
C_nk         = comb(n, k_exacto)

print(f"  P(X = {k_exacto}) = C({n},{k_exacto}) × {p:.4f}^{k_exacto} × {1-p:.4f}^{n-k_exacto}")
print(f"         = {C_nk} × {p**k_exacto:.6f} × {(1-p)**(n-k_exacto):.6f}")
print(f"         = {P_exacto:.6f}  ({P_exacto*100:.4f}%)")
print(f"\n  P(X ≥ 7) = P(X=7) + P(X=8) + P(X=9) + P(X=10)")
for k in range(7, n+1):
    print(f"            P(X={k}) = {binom.pmf(k, n, p):.6f}")
print(f"  P(X ≥ 7) = {P_al_menos_7:.6f}  ({P_al_menos_7*100:.4f}%)")

media_bin = n * p
var_bin   = n * p * (1-p)
print(f"\n  PARÁMETROS DE LA DISTRIBUCIÓN:")
print(f"  Media    μ = n×p         = {media_bin:.4f}")
print(f"  Varianza σ² = n×p×(1-p) = {var_bin:.4f}")
print(f"  Desv. Est. σ             = {var_bin**0.5:.4f}")

print(f"""
  INTERPRETACIÓN:
  La probabilidad de que exactamente {k_exacto} de las {n} observaciones
  seleccionadas aleatoriamente sean exportaciones es del
  {P_exacto*100:.4f}%. Con una probabilidad de éxito del {p*100:.2f}%,
  lo más probable es obtener alrededor de {media_bin:.1f} exportaciones
  (la media). La probabilidad de obtener al menos 7 exportaciones
  es del {P_al_menos_7*100:.4f}%, lo que indica que aunque es posible,
  no es lo más frecuente en el comercio exterior francés.
""")

tabla_bin = pd.DataFrame({
    'k': range(n+1),
    'P(X=k)': [binom.pmf(k, n, p) for k in range(n+1)],
    'P(X≤k)': [binom.cdf(k, n, p) for k in range(n+1)],
})
tabla_bin.to_csv(OUTPUT_DIR / "binomial.csv", index=False)
print("  ✓ Tabla guardada: binomial.csv")

# Gráfica Binomial
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ks      = range(n+1)
probs   = [binom.pmf(k, n, p) for k in ks]
colores = ['#C44E52' if k == k_exacto else '#55A868' if k >= 7 else '#4C72B0' for k in ks]

axes[0].bar(ks, probs, color=colores, alpha=0.85, edgecolor='white', linewidth=0.8)
for k, prob in zip(ks, probs):
    if prob > 0.01:
        axes[0].text(k, prob + 0.002, f'{prob:.4f}', ha='center', va='bottom', fontsize=7.5)
axes[0].set_title(f'Distribución Binomial B({n}, {p:.4f})\nComercio Exterior Francia', fontsize=11, fontweight='bold')
axes[0].set_xlabel('k (número de exportaciones)', fontsize=10)
axes[0].set_ylabel('P(X = k)', fontsize=10)
parches = [
    mpatches.Patch(color='#C44E52', alpha=0.85, label=f'P(X={k_exacto}) = {P_exacto:.4f}'),
    mpatches.Patch(color='#55A868', alpha=0.85, label=f'P(X≥7) = {P_al_menos_7:.4f}'),
    mpatches.Patch(color='#4C72B0', alpha=0.85, label='Otros valores'),
]
axes[0].legend(handles=parches, fontsize=9)

cdf_vals = [binom.cdf(k, n, p) for k in ks]
axes[1].step(ks, cdf_vals, color='#4C72B0', linewidth=2.5, where='post')
axes[1].fill_between(ks, cdf_vals, alpha=0.15, color='#4C72B0', step='post')
axes[1].axhline(binom.cdf(k_exacto, n, p), color='#C44E52', linestyle='--',
                linewidth=1.5, label=f'F({k_exacto}) = {binom.cdf(k_exacto,n,p):.4f}')
axes[1].set_title('Función de Distribución Acumulada\nBinomial', fontsize=11, fontweight='bold')
axes[1].set_xlabel('k', fontsize=10)
axes[1].set_ylabel('P(X ≤ k)', fontsize=10)
axes[1].legend(fontsize=9)
axes[1].set_xticks(ks)

plt.tight_layout()
plt.savefig(GRAFICAS / "14_binomial.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 14_binomial.png")

# ══════════════════════════════════════════════════════════════════════════
# 5.2 DISTRIBUCIÓN POISSON
# λ = promedio de países socios distintos por año con valor alto
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  5.2 DISTRIBUCIÓN POISSON")
print("="*60)

# Lambda: promedio de países socios NUEVOS por año en exportaciones
media_val = df_valido['valor'].mean()
df_export = df_valido[df_valido['flujo'] == 'X'].copy()

# Países socios por año con exportaciones de alto valor
paises_año = df_export[df_export['valor'] > media_val].groupby('año')['pais_socio'].nunique()
lam = round(paises_año.mean(), 2)
k_pois   = 8
k_menos5 = 4

print(f"""
  PROBLEMA:
  Francia tiene en promedio {lam} países socios con exportaciones
  de valor superior a la media ({media_val:,.0f} miles USD) por año.
  ¿Cuál es la probabilidad de que en un año determinado exactamente
  {k_pois} países socios registren exportaciones de alto valor?
  ¿Y de que sean menos de 5?

  PARÁMETROS:
  λ = {lam}  (promedio de países socios con export. alto valor/año)

  FÓRMULA:
  P(X = k) = (e^(-λ) × λ^k) / k!
""")

P_pois_exacto = poisson.pmf(k_pois, lam)
P_menos5      = poisson.cdf(k_menos5, lam)

print(f"  P(X = {k_pois}) = (e^(-{lam}) × {lam}^{k_pois}) / {k_pois}!")
print(f"          = {exp(-lam):.6f} × {lam**k_pois:.4f} / {factorial(k_pois):,}")
print(f"          = {P_pois_exacto:.6f}  ({P_pois_exacto*100:.4f}%)")
print(f"\n  P(X < 5) = P(X ≤ 4) = {P_menos5:.6f}  ({P_menos5*100:.4f}%)")
for k in range(5):
    print(f"            P(X={k}) = {poisson.pmf(k, lam):.6f}")

print(f"\n  PARÁMETROS DE LA DISTRIBUCIÓN:")
print(f"  Media    μ = λ  = {lam}")
print(f"  Varianza σ² = λ = {lam}")
print(f"  Desv. Est. σ    = {lam**0.5:.4f}")

print(f"""
  INTERPRETACIÓN:
  Con un promedio de {lam} países socios por año que registran
  exportaciones francesas de alto valor, la probabilidad de
  que en un año específico exactamente {k_pois} países socios
  tengan este nivel de intercambio es del {P_pois_exacto*100:.4f}%.
  La probabilidad de que sean menos de 5 países es del
  {P_menos5*100:.4f}%, lo que indica que Francia mantiene
  consistentemente un número elevado de socios comerciales
  estratégicos en sus exportaciones de mayor volumen.
""")

k_max = int(lam * 2.5)
tabla_pois = pd.DataFrame({
    'k': range(k_max+1),
    'P(X=k)': [poisson.pmf(k, lam) for k in range(k_max+1)],
    'P(X≤k)': [poisson.cdf(k, lam) for k in range(k_max+1)],
})
tabla_pois.to_csv(OUTPUT_DIR / "poisson.csv", index=False)
print("  ✓ Tabla guardada: poisson.csv")

# Gráfica Poisson
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ks_p    = range(k_max+1)
probs_p = [poisson.pmf(k, lam) for k in ks_p]
colores_p = ['#C44E52' if k == k_pois else '#55A868' if k <= k_menos5 else '#4C72B0' for k in ks_p]

axes[0].bar(ks_p, probs_p, color=colores_p, alpha=0.85, edgecolor='white', linewidth=0.6)
axes[0].axvline(lam, color='black', linewidth=2, linestyle='--', label=f'λ = {lam}')
for k, prob in zip(ks_p, probs_p):
    if prob > 0.02:
        axes[0].text(k, prob + 0.002, f'{prob:.3f}', ha='center', va='bottom', fontsize=7.5)
axes[0].set_title(f'Distribución Poisson (λ = {lam})\nPaíses Socios con Export. Alto Valor — Francia',
                  fontsize=11, fontweight='bold')
axes[0].set_xlabel('k (países socios con exportaciones de alto valor)', fontsize=10)
axes[0].set_ylabel('P(X = k)', fontsize=10)
parches_p = [
    mpatches.Patch(color='#C44E52', alpha=0.85, label=f'P(X={k_pois}) = {P_pois_exacto:.4f}'),
    mpatches.Patch(color='#55A868', alpha=0.85, label=f'P(X<5) = {P_menos5:.4f}'),
    mpatches.Patch(color='#4C72B0', alpha=0.85, label='Otros valores'),
]
axes[0].legend(handles=parches_p, fontsize=9)

from scipy.stats import norm
x_norm    = np.linspace(0, k_max, 200)
norm_vals = norm.pdf(x_norm, lam, lam**0.5)
axes[1].bar(ks_p, probs_p, color='#4C72B0', alpha=0.5, edgecolor='white', label='Poisson')
axes[1].plot(x_norm, norm_vals, color='#C44E52', linewidth=2.5,
             label=f'Normal aprox.\nμ={lam}, σ={lam**0.5:.2f}')
axes[1].set_title('Poisson vs Aproximación Normal\nFrancia — Países Socios Alto Valor',
                  fontsize=11, fontweight='bold')
axes[1].set_xlabel('k', fontsize=10)
axes[1].set_ylabel('Probabilidad', fontsize=10)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(GRAFICAS / "15_poisson.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 15_poisson.png")

# ══════════════════════════════════════════════════════════════════════════
# 5.3 DISTRIBUCIÓN HIPERGEOMÉTRICA
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  5.3 DISTRIBUCIÓN HIPERGEOMÉTRICA")
print("="*60)

# Usar france_usd con tamaño de empresa disponible
df_hip  = df_valido.copy()
N_hip   = len(df_hip)
K_hip   = len(df_hip[(df_hip['flujo'] == 'X') & (df_hip['tamano_empresa'] == 'S_GE250')])
n_hip   = 20
k_hip   = 3

print(f"""
  PROBLEMA:
  Del total de {N_hip:,} observaciones válidas de comercio exterior
  de Francia, {K_hip:,} son exportaciones de empresas grandes
  (≥250 empleados). Un auditor selecciona aleatoriamente
  {n_hip} observaciones SIN reemplazo.
  ¿Cuál es la probabilidad de que exactamente {k_hip} sean
  exportaciones de empresas grandes?

  PARÁMETROS:
  N = {N_hip:,}   (población total)
  K = {K_hip:,}    (exportaciones de empresas grandes)
  n = {n_hip}     (muestra seleccionada)
  k = {k_hip}      (éxitos buscados)

  FÓRMULA:
  P(X = k) = C(K,k) × C(N-K, n-k) / C(N,n)
""")

P_hip_exacto  = hypergeom.pmf(k_hip, N_hip, K_hip, n_hip)
P_al_menos3   = 1 - hypergeom.cdf(k_hip - 1, N_hip, K_hip, n_hip)
C_Kk          = comb(K_hip, k_hip)
C_NKnk        = comb(N_hip - K_hip, n_hip - k_hip)
C_Nn          = comb(N_hip, n_hip)

print(f"  P(X = {k_hip}) = C({K_hip},{k_hip}) × C({N_hip-K_hip},{n_hip-k_hip}) / C({N_hip},{n_hip})")
print(f"          = {C_Kk:,} × {C_NKnk:,}")
print(f"            ÷ {C_Nn:,}")
print(f"          = {P_hip_exacto:.6f}  ({P_hip_exacto*100:.4f}%)")
print(f"\n  P(X ≥ {k_hip}) = {P_al_menos3:.6f}  ({P_al_menos3*100:.4f}%)")

media_hip = n_hip * K_hip / N_hip
var_hip   = n_hip * (K_hip/N_hip) * ((N_hip-K_hip)/N_hip) * ((N_hip-n_hip)/(N_hip-1))

print(f"\n  PARÁMETROS DE LA DISTRIBUCIÓN:")
print(f"  Media    μ = n×K/N = {media_hip:.4f}")
print(f"  Varianza σ²        = {var_hip:.4f}")
print(f"  Desv. Est. σ       = {var_hip**0.5:.4f}")

print(f"""
  INTERPRETACIÓN:
  Al seleccionar {n_hip} observaciones sin reemplazo del dataset
  de Francia, la probabilidad de encontrar exactamente {k_hip}
  exportaciones de empresas grandes es del {P_hip_exacto*100:.4f}%.
  En promedio se esperan {media_hip:.2f} exportaciones de empresas
  grandes por cada muestra de {n_hip} observaciones.
  La probabilidad de encontrar {k_hip} o más es del
  {P_al_menos3*100:.4f}%, lo que refleja la participación
  significativa de las grandes empresas francesas en el
  comercio exterior del país.
""")

k_range = range(min(K_hip, n_hip) + 1)
tabla_hip = pd.DataFrame({
    'k': k_range,
    'P(X=k)': [hypergeom.pmf(k, N_hip, K_hip, n_hip) for k in k_range],
    'P(X≤k)': [hypergeom.cdf(k, N_hip, K_hip, n_hip) for k in k_range],
})
tabla_hip.to_csv(OUTPUT_DIR / "hipergeometrica.csv", index=False)
print("  ✓ Tabla guardada: hipergeometrica.csv")

# Gráfica Hipergeométrica
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ks_h    = list(k_range)
probs_h = [hypergeom.pmf(k, N_hip, K_hip, n_hip) for k in ks_h]
colores_h = ['#C44E52' if k == k_hip else '#55A868' if k >= k_hip else '#4C72B0' for k in ks_h]

axes[0].bar(ks_h, probs_h, color=colores_h, alpha=0.85, edgecolor='white', linewidth=0.8)
axes[0].axvline(media_hip, color='black', linewidth=2, linestyle='--', label=f'μ = {media_hip:.2f}')
for k, prob in zip(ks_h, probs_h):
    if prob > 0.005:
        axes[0].text(k, prob + 0.002, f'{prob:.4f}', ha='center', va='bottom', fontsize=7.5)
axes[0].set_title(f'Distribución Hipergeométrica\nN={N_hip:,}, K={K_hip}, n={n_hip}',
                  fontsize=11, fontweight='bold')
axes[0].set_xlabel('k (exportaciones de empresas grandes)', fontsize=10)
axes[0].set_ylabel('P(X = k)', fontsize=10)
parches_h = [
    mpatches.Patch(color='#C44E52', alpha=0.85, label=f'P(X={k_hip}) = {P_hip_exacto:.4f}'),
    mpatches.Patch(color='#55A868', alpha=0.85, label=f'P(X≥{k_hip}) = {P_al_menos3:.4f}'),
    mpatches.Patch(color='#4C72B0', alpha=0.85, label='Otros valores'),
]
axes[0].legend(handles=parches_h, fontsize=9)

p_aprox   = K_hip / N_hip
probs_bin = [binom.pmf(k, n_hip, p_aprox) for k in ks_h]
axes[1].bar(ks_h, probs_h, color='#4C72B0', alpha=0.5, edgecolor='white',
            label='Hipergeométrica', width=0.4, align='edge')
axes[1].bar([k+0.4 for k in ks_h], probs_bin, color='#DD8452', alpha=0.5,
            edgecolor='white', label=f'Binomial aprox.\np={p_aprox:.4f}', width=0.4, align='edge')
axes[1].set_title('Hipergeométrica vs Binomial Aproximada\nFrancia — Exportaciones Emp. Grandes',
                  fontsize=11, fontweight='bold')
axes[1].set_xlabel('k', fontsize=10)
axes[1].set_ylabel('Probabilidad', fontsize=10)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(GRAFICAS / "16_hipergeometrica.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 16_hipergeometrica.png")

# ══════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  RESUMEN — DISTRIBUCIONES DISCRETAS")
print("="*60)
print(f"\n  BINOMIAL B({n}, {p:.4f}):")
print(f"  P(X = {k_exacto})  = {P_exacto:.6f}  ({P_exacto*100:.4f}%)")
print(f"  P(X ≥ 7)   = {P_al_menos_7:.6f}  ({P_al_menos_7*100:.4f}%)")
print(f"  Media      = {media_bin:.4f}")
print(f"\n  POISSON (λ = {lam}):")
print(f"  P(X = {k_pois})    = {P_pois_exacto:.6f}  ({P_pois_exacto*100:.4f}%)")
print(f"  P(X < 5)   = {P_menos5:.6f}  ({P_menos5*100:.4f}%)")
print(f"  Media      = {lam}")
print(f"\n  HIPERGEOMÉTRICA (N={N_hip:,}, K={K_hip}, n={n_hip}):")
print(f"  P(X = {k_hip})    = {P_hip_exacto:.6f}  ({P_hip_exacto*100:.4f}%)")
print(f"  P(X ≥ {k_hip})   = {P_al_menos3:.6f}  ({P_al_menos3*100:.4f}%)")
print(f"  Media      = {media_hip:.4f}")
print("\n  ✅ Fase 6 completada.")
print("  Revisa las carpetas /graficas/ y /output/")