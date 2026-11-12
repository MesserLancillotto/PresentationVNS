import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 1. Definizione dei 5 punti per il calcolo della curva
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y = np.array([4.0, 1.0, 3.0, 0.0, 6.0])

# Coordinate x in cui inserire le linee verticali rosse
x_linee_verticali = [0.5, 2.8]

# Calcolo della curva tramite spline cubica passante per i punti
cs = CubicSpline(x, y)
x_dense = np.linspace(x[0], x[-1], 500)
y_dense = cs(x_dense)

# 8. Calcolo dei soli minimi locali (derivata prima = 0 e derivata seconda > 0)
d1 = cs.derivative(1)
d2 = cs.derivative(2)
roots = d1.roots()

# Filtro per mantenere solo i punti nell'intervallo con derivata seconda positiva
mask = (roots >= x[0]) & (roots <= x[-1]) & (d2(roots) > 0)
x_custom = roots[mask]
y_custom = cs(x_custom)

# 2. Configurazione del grafico su sfondo bianco
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Disegno della curva blu (6. I punti originali della curva non vengono tracciati)
ax.plot(x_dense, y_dense, color='blue', linewidth=2)

# 7. & 8. Disegno dei soli punti di minimo locale
if len(x_custom) > 0:
    ax.scatter(x_custom, y_custom, color='black', s=50, zorder=5)
ax.scatter(2.8, cs(2.8), color='black', s=50, zorder=5)

# 3. Linee verticali rosse tratteggiate nei punti indicati
for xv in x_linee_verticali:
    ax.axvline(x=xv, color='red', linestyle='--', linewidth=1.5)

# 4. Rimozione completa dell'asse y
ax.spines['left'].set_visible(False)
ax.yaxis.set_visible(False)

# 5. Rimozione dei numeri (etichette) dall'asse x
ax.xaxis.set_major_formatter(plt.NullFormatter())

plt.tight_layout()

# Salvataggio dell'immagine su file
plt.savefig('curva_punti.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')

plt.close()