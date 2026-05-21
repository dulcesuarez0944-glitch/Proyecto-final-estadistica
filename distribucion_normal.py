"""
=====================================================================
PROYECTO FINAL - ESTADÍSTICA
Fase 7 — Distribuciones de Probabilidad con Variable Aleatoria Continua
País: Francia | Fuente: OCDE TEC ISIC4
=====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm, shapiro, kstest
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
datos     = df_valido['valor'].dropna()

mu    = datos.mean()
sigma = datos.std(ddof=1)

print(f"Observaciones válidas : {len(datos):,}")
print(f"Media (μ)             : {mu:,.2f} miles USD")
print(f"Desv. Est. (σ)        : {sigma:,.2f} miles USD")

# ══════════════════════════════════════════════════════════════════════════
# 6.1 DISTRIBUCIÓN NORMAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  6.1 DISTRIBUCIÓN NORMAL")
print("="*60)

x1 = 10_000_000
x2_inf = 1_000_000
x2_sup = 20_000_000

print(f"""
  PROBLEMA:
  Los valores comerciales de Francia se modelan con una
  distribución normal con:
  μ = {mu:,.2f} miles USD
  σ = {sigma:,.2f} miles USD

  PREGUNTA A: ¿Cuál es la probabilidad de que un valor
  comercial sea menor a {x1:,} miles USD?

  PREGUNTA B: ¿Cuál es la probabilidad de que esté entre
  {x2_inf:,} y {x2_sup:,} miles USD?

  FÓRMULA:
  X ~ N(μ, σ²)
  P(X < x) = Φ((x - μ) / σ)
""")

# Pregunta A
P_menor_x1 = norm.cdf(x1, mu, sigma)
Z_x1       = (x1 - mu) / sigma
print(f"  PREGUNTA A:")
print(f"  Z = (x - μ) / σ = ({x1:,} - {mu:,.2f}) / {sigma:,.2f}")
print(f"  Z = {Z_x1:.4f}")
print(f"  P(X < {x1:,}) = Φ({Z_x1:.4f}) = {P_menor_x1:.6f}  ({P_menor_x1*100:.4f}%)")

# Pregunta B
P_entre    = norm.cdf(x2_sup, mu, sigma) - norm.cdf(x2_inf, mu, sigma)
Z_inf      = (x2_inf - mu) / sigma
Z_sup      = (x2_sup - mu) / sigma
print(f"\n  PREGUNTA B:")
print(f"  Z_inf = ({x2_inf:,} - {mu:,.2f}) / {sigma:,.2f} = {Z_inf:.4f}")
print(f"  Z_sup = ({x2_sup:,} - {mu:,.2f}) / {sigma:,.2f} = {Z_sup:.4f}")
print(f"  P({x2_inf:,} < X < {x2_sup:,}) = Φ({Z_sup:.4f}) - Φ({Z_inf:.4f})")
print(f"  = {norm.cdf(x2_sup,mu,sigma):.6f} - {norm.cdf(x2_inf,mu,sigma):.6f}")
print(f"  = {P_entre:.6f}  ({P_entre*100:.4f}%)")

# Prueba de normalidad
stat_sh, p_sh = shapiro(datos.sample(min(5000, len(datos)), random_state=42))
print(f"\n  PRUEBA DE NORMALIDAD (Shapiro-Wilk):")
print(f"  Estadístico W = {stat_sh:.4f}")
print(f"  p-valor       = {p_sh:.6f}")
print(f"  {'Los datos siguen una distribución normal' if p_sh > 0.05 else 'Los datos no siguen estrictamente una normal (distribución sesgada)'}")

print(f"""
  INTERPRETACIÓN:
  Modelando el valor comercial de Francia como una variable
  aleatoria continua con distribución normal N({mu:,.0f}, {sigma:,.0f}²),
  la probabilidad de que una operación comercial sea menor
  a {x1:,} miles USD es del {P_menor_x1*100:.4f}%. Esto refleja
  que la gran mayoría de intercambios comerciales de Francia
  se concentran en valores relativamente bajos, mientras que
  los intercambios de gran escala son mucho menos frecuentes.
  La probabilidad de encontrar un valor entre {x2_inf:,} y
  {x2_sup:,} miles USD es del {P_entre*100:.4f}%.
""")

# Guardar tabla
x_vals = [500_000, 1_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000]
tabla_norm = pd.DataFrame({
    'x (miles USD)': x_vals,
    'Z': [(x - mu)/sigma for x in x_vals],
    'P(X < x)': [norm.cdf(x, mu, sigma) for x in x_vals],
    'P(X > x)': [1 - norm.cdf(x, mu, sigma) for x in x_vals],
})
tabla_norm.to_csv(OUTPUT_DIR / "normal.csv", index=False)
print("  ✓ Tabla guardada: normal.csv")

# ── Gráfica 1: Curva Normal con áreas sombreadas ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

x_range = np.linspace(mu - 3.5*sigma, mu + 3.5*sigma, 500)
y_range = norm.pdf(x_range, mu, sigma)

# Área P(X < x1)
x_fill1 = np.linspace(mu - 3.5*sigma, x1, 300)
axes[0].plot(x_range, y_range, color='#4C72B0', linewidth=2.5, label='N(μ, σ²)')
axes[0].fill_between(x_fill1, norm.pdf(x_fill1, mu, sigma),
                     alpha=0.4, color='#55A868', label=f'P(X<{x1/1e6:.0f}M) = {P_menor_x1:.4f}')
axes[0].axvline(x1,  color='#55A868', linewidth=2, linestyle='--')
axes[0].axvline(mu,  color='#C44E52', linewidth=2, linestyle='-', label=f'μ = {mu/1e6:.2f}M')

# Área P(x2_inf < X < x2_sup)
x_fill2 = np.linspace(x2_inf, x2_sup, 300)
axes[0].fill_between(x_fill2, norm.pdf(x_fill2, mu, sigma),
                     alpha=0.4, color='#DD8452', label=f'P({x2_inf/1e6:.0f}M<X<{x2_sup/1e6:.0f}M) = {P_entre:.4f}')

axes[0].set_title('Distribución Normal — Valor Comercial Francia\nÁreas de Probabilidad',
                  fontsize=11, fontweight='bold')
axes[0].set_xlabel('Valor Comercial (miles de USD)', fontsize=10)
axes[0].set_ylabel('Densidad de probabilidad', fontsize=10)
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
axes[0].legend(fontsize=8)

# Histograma vs curva normal ajustada
axes[1].hist(datos, bins=80, density=True, color='#4C72B0', alpha=0.5,
             edgecolor='white', linewidth=0.4, label='Datos reales')
axes[1].plot(x_range, y_range, color='#C44E52', linewidth=2.5,
             label=f'Normal ajustada\nμ={mu/1e6:.2f}M, σ={sigma/1e6:.2f}M')
axes[1].set_title('Histograma vs Curva Normal Ajustada\nValor Comercial Francia',
                  fontsize=11, fontweight='bold')
axes[1].set_xlabel('Valor Comercial (miles de USD)', fontsize=10)
axes[1].set_ylabel('Densidad', fontsize=10)
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
axes[1].legend(fontsize=9)
axes[1].set_xlim(mu - 2*sigma, mu + 2*sigma)

plt.tight_layout()
plt.savefig(GRAFICAS / "17_distribucion_normal.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 17_distribucion_normal.png")

# ══════════════════════════════════════════════════════════════════════════
# 6.2 DISTRIBUCIÓN NORMAL ESTÁNDAR
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  6.2 DISTRIBUCIÓN NORMAL ESTÁNDAR")
print("="*60)

x_z = 50_000_000
Z_objetivo = (x_z - mu) / sigma

print(f"""
  PROBLEMA:
  Estandarizando los valores comerciales de Francia:
  Z = (X - μ) / σ = (X - {mu:,.2f}) / {sigma:,.2f}

  PREGUNTA A: ¿Cuál es el valor Z de un intercambio
  de {x_z:,} miles USD?

  PREGUNTA B: Regla empírica 68-95-99.7
  ¿Qué porcentaje de intercambios está dentro de
  1σ, 2σ y 3σ de la media?
""")

print(f"  PREGUNTA A:")
print(f"  Z = ({x_z:,} - {mu:,.2f}) / {sigma:,.2f}")
print(f"  Z = {Z_objetivo:.4f}")
print(f"  P(X < {x_z:,}) = P(Z < {Z_objetivo:.4f}) = {norm.cdf(Z_objetivo):.6f}  ({norm.cdf(Z_objetivo)*100:.4f}%)")

# Regla empírica
P_1sigma = norm.cdf(1) - norm.cdf(-1)
P_2sigma = norm.cdf(2) - norm.cdf(-2)
P_3sigma = norm.cdf(3) - norm.cdf(-3)

# Valores reales en los datos
n1 = ((datos >= mu - sigma)   & (datos <= mu + sigma)).sum()
n2 = ((datos >= mu - 2*sigma) & (datos <= mu + 2*sigma)).sum()
n3 = ((datos >= mu - 3*sigma) & (datos <= mu + 3*sigma)).sum()
total_n = len(datos)

print(f"\n  PREGUNTA B — REGLA EMPÍRICA:")
print(f"  {'Rango':<30} {'Teórico':>10} {'Real (Francia)':>15}")
print(f"  {'-'*55}")
print(f"  {'μ ± 1σ':<30} {P_1sigma*100:>9.2f}% {n1/total_n*100:>14.2f}%  ({n1:,} obs)")
print(f"  {'μ ± 2σ':<30} {P_2sigma*100:>9.2f}% {n2/total_n*100:>14.2f}%  ({n2:,} obs)")
print(f"  {'μ ± 3σ':<30} {P_3sigma*100:>9.2f}% {n3/total_n*100:>14.2f}%  ({n3:,} obs)")

# Tabla Z estándar
z_vals = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
tabla_z = pd.DataFrame({
    'Z': z_vals,
    'Φ(Z) = P(Z ≤ z)': [norm.cdf(z) for z in z_vals],
    'P(Z > z)': [1 - norm.cdf(z) for z in z_vals],
    'X correspondiente (miles USD)': [mu + z*sigma for z in z_vals],
})
tabla_z.to_csv(OUTPUT_DIR / "normal_estandar.csv", index=False)
print(f"\n  ✓ Tabla guardada: normal_estandar.csv")

print(f"""
  INTERPRETACIÓN:
  Un valor comercial de {x_z:,} miles USD corresponde a un
  puntaje Z de {Z_objetivo:.4f}, lo que significa que está
  {abs(Z_objetivo):.2f} desviaciones estándar {'por encima' if Z_objetivo > 0 else 'por debajo'}
  de la media. El {norm.cdf(Z_objetivo)*100:.2f}% de los intercambios
  comerciales de Francia se encuentran por debajo de este valor.

  Aplicando la regla empírica, el {P_1sigma*100:.2f}% de los
  intercambios deberían estar dentro de una desviación estándar
  de la media. En los datos reales de Francia, este porcentaje
  es del {n1/total_n*100:.2f}%, lo que confirma que la distribución
  del comercio exterior francés presenta un sesgo positivo
  importante, concentrando la mayoría de operaciones en
  valores bajos con algunas operaciones de muy alto valor.
""")

# ── Gráfica 2: Curva Normal Estándar ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

z_range = np.linspace(-4, 4, 500)
y_std   = norm.pdf(z_range)

# Curva Z con área objetivo
axes[0].plot(z_range, y_std, color='#4C72B0', linewidth=2.5, label='N(0,1)')
x_fill_z = np.linspace(-4, Z_objetivo, 300)
axes[0].fill_between(x_fill_z, norm.pdf(x_fill_z),
                     alpha=0.4, color='#55A868',
                     label=f'P(Z<{Z_objetivo:.2f}) = {norm.cdf(Z_objetivo):.4f}')
axes[0].axvline(Z_objetivo, color='#55A868', linewidth=2, linestyle='--')
axes[0].axvline(0, color='#C44E52', linewidth=1.5, linestyle='-', label='μ = 0')

axes[0].set_title(f'Distribución Normal Estándar Z\nValor objetivo: X = {x_z/1e6:.0f}M → Z = {Z_objetivo:.4f}',
                  fontsize=11, fontweight='bold')
axes[0].set_xlabel('Z', fontsize=10)
axes[0].set_ylabel('Densidad f(z)', fontsize=10)
axes[0].legend(fontsize=9)

# Regla empírica visual
colores_sigma = ['#4C72B0', '#55A868', '#DD8452']
alphas_sigma  = [0.5, 0.35, 0.2]
sigmas        = [1, 2, 3]
probs_sigma   = [P_1sigma, P_2sigma, P_3sigma]

axes[1].plot(z_range, y_std, color='black', linewidth=2)
for i, (s, prob, color, alpha) in enumerate(zip(sigmas, probs_sigma, colores_sigma, alphas_sigma)):
    x_s = np.linspace(-s, s, 300)
    axes[1].fill_between(x_s, norm.pdf(x_s), alpha=alpha, color=color,
                         label=f'μ ± {s}σ: {prob*100:.2f}%')

axes[1].axvline(0, color='#C44E52', linewidth=2, linestyle='--', label='μ = 0')
axes[1].set_title('Regla Empírica 68 - 95 - 99.7\nDistribución Normal Estándar',
                  fontsize=11, fontweight='bold')
axes[1].set_xlabel('Z (desviaciones estándar)', fontsize=10)
axes[1].set_ylabel('Densidad f(z)', fontsize=10)
axes[1].set_xticks([-3, -2, -1, 0, 1, 2, 3])
axes[1].set_xticklabels(['-3σ', '-2σ', '-1σ', 'μ', '+1σ', '+2σ', '+3σ'])
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(GRAFICAS / "18_normal_estandar.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 18_normal_estandar.png")

# ── Gráfica 3: QQ Plot ─────────────────────────────────────────────────────
from scipy.stats import probplot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

probplot(datos, dist="norm", plot=axes[0])
axes[0].set_title('QQ Plot — Valor Comercial Francia\nvs Distribución Normal Teórica',
                  fontsize=11, fontweight='bold')
axes[0].get_lines()[0].set(color='#4C72B0', alpha=0.5, markersize=3)
axes[0].get_lines()[1].set(color='#C44E52', linewidth=2)

# Datos estandarizados
datos_z = (datos - mu) / sigma
axes[1].hist(datos_z, bins=80, density=True, color='#4C72B0', alpha=0.6,
             edgecolor='white', linewidth=0.4, label='Datos estandarizados')
axes[1].plot(z_range, norm.pdf(z_range), color='#C44E52', linewidth=2.5,
             label='N(0,1) teórica')
axes[1].set_title('Datos Estandarizados de Francia\nvs Curva Normal Estándar',
                  fontsize=11, fontweight='bold')
axes[1].set_xlabel('Z', fontsize=10)
axes[1].set_ylabel('Densidad', fontsize=10)
axes[1].set_xlim(-4, 4)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(GRAFICAS / "19_qqplot_normal.png", bbox_inches='tight')
plt.close()
print("  ✓ Gráfica guardada: 19_qqplot_normal.png")

# ══════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  RESUMEN — DISTRIBUCIÓN NORMAL Y NORMAL ESTÁNDAR")
print("="*60)
print(f"\n  Parámetros:")
print(f"  μ = {mu:,.2f} miles USD")
print(f"  σ = {sigma:,.2f} miles USD")
print(f"\n  DISTRIBUCIÓN NORMAL:")
print(f"  P(X < {x1/1e6:.0f}M)              = {P_menor_x1:.6f}  ({P_menor_x1*100:.4f}%)")
print(f"  P({x2_inf/1e6:.0f}M < X < {x2_sup/1e6:.0f}M)    = {P_entre:.6f}  ({P_entre*100:.4f}%)")
print(f"\n  DISTRIBUCIÓN NORMAL ESTÁNDAR:")
print(f"  Z para X = {x_z/1e6:.0f}M           = {Z_objetivo:.4f}")
print(f"  P(Z < {Z_objetivo:.4f})            = {norm.cdf(Z_objetivo):.6f}  ({norm.cdf(Z_objetivo)*100:.4f}%)")
print(f"\n  REGLA EMPÍRICA:")
print(f"  μ ± 1σ  → Teórico: 68.27%  | Real: {n1/total_n*100:.2f}%")
print(f"  μ ± 2σ  → Teórico: 95.45%  | Real: {n2/total_n*100:.2f}%")
print(f"  μ ± 3σ  → Teórico: 99.73%  | Real: {n3/total_n*100:.2f}%")
print("\n  ✅ Fase 7 completada.")
print("  Revisa las carpetas /graficas/ y /output/")