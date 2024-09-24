import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


resultados=np.genfromtxt("resultados.dat")
vx=np.genfromtxt("velocidades.dat")[0]
vy=np.genfromtxt("velocidades.dat")[1]
vx2=abs(vx)#VELOCIDAD CUADRATICA EN X
vy2=abs(vy)#VELOCIDAD CUADRATICA EN X
vpromx= np.nanmean(vx) # VELOCIDAD PROMEDIO EN X
vpromy=np.nanmean(vy)  # VELOCIDAD PROMEDIO EN Y
vpromx2=np.nanmean(vx2) # VELOCIDAD CUADRATICA PROMEDIO EN X
vpromy2=np.nanmean(vy2) # VELOCIDAD CUADRATICA PROMEDIO EN Y

# Generación de Histogramas
# VELOCIDAD EN Y
plt.figure(figsize=(10, 5))
plt.title('Histograma velocidades en Y')
VY= np.linspace( - 4*vpromy2,   4*vpromy2, 1000)
DATAVY= norm.pdf(VY, 0, vpromy2)
plt.plot(VY, DATAVY, color='darkslateblue', linewidth=3)
plt.hist(vy, bins=50, density=True,color="mediumslateblue", alpha=0.9)
plt.grid()
plt.savefig("H-VY.png")

# VELOCIDAD EN x
plt.figure(figsize=(10, 5))
plt.title('Histograma velocidades en X')
#Agrega una linea vertical en la velocidad de deriva
plt.axvline(x=vpromx, color='indianred', linestyle='--', linewidth=3, label='Velocidad de deriva')
VX= np.linspace( vpromx- 4*vpromx2, vpromx+4*vpromx2, 1000)
DATAVX= norm.pdf(VX, vpromx, vpromx2)
plt.plot(VX, DATAVX, 'firebrick', linewidth=2)
plt.hist(vx, bins=50, density=True,color="coral", alpha=0.9)
plt.legend()
plt.grid()
plt.savefig("H-VX.png")

# DISTRIBUCIÓN MAXWELL-BOLTZMAN
# Calcular la magnitud de la velocidad total
v = np.sqrt(vx**2 + vy**2)
plt.figure(figsize=(10, 6))
plt.hist(v, bins=50, density=True, color='mediumslateblue', alpha=0.9, label='Histograma de Velocidades')
plt.title('Histograma de Velocidades Totales')
plt.xlabel('Velocidad (v)')
plt.ylabel('Densidad de Probabilidad')
plt.legend()
plt.grid()
plt.savefig("H-V.png")
