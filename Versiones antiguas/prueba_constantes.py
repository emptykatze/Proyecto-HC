import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
from scipy import constants

# Definimos las constantes
kb = constants.Boltzmann * 10**24  # Convertimos la constante de Boltzmann
me = constants.electron_mass  # Masa del electrón
T = 1  # Temperatura en Kelvin

# Generar datos de la distribución de Maxwell-Boltzmann
scale = np.sqrt(kb * T / me)  # La escala para la distribución
data = maxwell.rvs(scale=scale, size=10900)  # Generamos 10,000 muestras

# Crear el histograma normalizado
plt.hist(data, bins=30, density=True, color='lightblue', edgecolor='black')

# Etiquetas
plt.title('Distribución de Maxwell-Boltzmann')
plt.xlabel('Velocidad')
plt.ylabel('Densidad')

# Mostrar el gráfico
plt.show()
#dt=1*10**(-18)
#tiempo_total =1*10**(-14)  # Tiempo total del movimiento (segundos)
#num_puntos = int(tiempo_total/dt)  # Número de puntos en el dataframe  1000
#print(num_puntos)
