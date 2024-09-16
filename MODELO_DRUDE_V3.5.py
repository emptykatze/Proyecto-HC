import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import random
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from scipy import constants
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
"""
Las medidas que utilizamos son:
distancia: picometros- 235pm de radio (creo) de electrones.
para la carga y la masa del electrón los normalizamos siendo 1 culomb y 1kg.
debido a lo anterior debemos convertir las medidas del campo magnético y la constante de Boltzmann:
V/m=5.6797X10^48pm/s^2.
La constante de Boltzmann es k=1.25639059X10^33pm^2/(ks^2)
"""
#Definir clase particula
class particula:
    def __init__(self, x, y, z, vx, vy, vz, radio=100,m=1,carga=-constants.elementary_charge,movimiento=True,color="purple"):
        self.r = np.array([x, y,z])
        self.v = np.array([vx, vy, vz])
        self.radio=radio
        self.m = m
        self.px = []
        self.py = []
        self.pz = []
        self.q=carga
        self.color=color
        self.movimiento=movimiento
        self.tau=[0]
    
    def mov(self, dt):
        self.px.append(self.r[0])
        self.py.append(self.r[1])
        self.pz.append(self.r[2])
        if self.movimiento==True:
            self.r=self.r+self.v*dt
        
    def colision(self, particula2=None,t=0):        
        # Verifica si las particulas se tocan:
        D2 = (particula2.r[0] - self.r[0])*(particula2.r[0] - self.r[0]) + (particula2.r[1] - self.r[1])*(particula2.r[1] - self.r[1])+(particula2.r[2] - self.r[2])*(particula2.r[2] - self.r[2])
        suma_radios = self.radio + particula2.radio
        # separacion entre particulas:
        d=np.array(particula2.r)-np.array(self.r)
        # Velocidad relativa entre particulas:
        vrel=particula2.v-self.v
        if self!=particula2 and D2 <= suma_radios*suma_radios and np.dot(vrel,d)<0 and (particula2.movimiento==False or self.movimiento==False):
            self.v=-self.v 
            particula2.v=-particula2.v  
            # Si hay un choque se va a guardar el tiempo en el que ocurre menos el tiempo anterior
            
            self.tau.append(t-self.tau[-1])
            particula2.tau.append(t-particula2.tau[-1])

    def campo_electrico(self,E,dt):
        a=self.q*E/self.m
        self.v=a*dt+self.v
        
        
        



# Definimos un campo electrico en dirección x:
E=np.array([1*10**(-10),0.,0.])




# CREACIÓN DEL DATA FRAME
# Definir los parámetros del movimiento
dt=0.005
tiempo_total = 5  # Tiempo total del movimiento (segundos)
num_puntos = int(tiempo_total/dt)  # Número de puntos en el dataframe  1000
#dt=tiempo_total/num_puntos
time = np.linspace(0, tiempo_total,num_puntos)

# LIMITES DE LA CAJA
limx=20
limy=20
limz=20
# NUMERO DE IONES MXM
M=5
M2=M**3
# NUMERO DE ELECTRONES
N=M2
#Lista de particulas:
particulas=[]
#Definir radio de electrones, protones y distancia entre protones en la red:
r_e, m_e=.4, constants.m_e
r_p, m_p=1.2, constants.m_p

# Definir cuadricula de iones, o atomos y agregarlos a la lista de particula
for i in range(M):
    for j in range(M):
        for k in range (M):
            particulas.append(particula((-limx+r_p*1.1)+i*5,(-limy+r_p*1.1)+j*5,(-limz+r_p*1.1)+k*5,0,0,0,r_p,m_p,color="red",movimiento=False))
# Definir electrones y agregarlos a la lista particulas, se tiene que tener en cuenta que no se pueden solapar entre ellos ni con los radios de los protones
# Entonces si se genera alguno que se solape con los demas, se genera otra posicion 
for i in range(N):
    gx, gy , gz=random.uniform(-limx+r_e,limx-r_e),random.uniform(-limy+r_e,limy-r_e),random.uniform(-limz+r_e,limz-r_e)
    # Generar nuevas posiciones gx y gy aleatoriamente hasta que no solapen con ninguna partícula
    for i in particulas:
        solapado = True
        while solapado:
            gx = random.uniform(-limx + r_e, limx - r_e)
            gy = random.uniform(-limy + r_e, limy - r_e)
            gz = random.uniform(-limz + r_e, limz - r_e)
            # Verificar si la nueva posición (gx, gy) no solapa con ninguna partícula
            solapado = False
            for p in particulas:
                if p != i:  # No verificar contra sí misma
                    if ((gx-p.r[0])**2+(gy-p.r[1])**2+(gz-p.r[1])**2)**0.5 < (r_e + p.radio):
                        solapado = True
                        break
    gvx, gvy, gvz=random.uniform(-limx,limx),random.uniform(-limy,limy)*15,random.uniform(-limy,limy)*15
    #gvx, gvy=0,0
    particulas.append(particula(gx,gy,gz,gvx,gvy,gvz,r_e,m_e,color="purple"))


contador=0
for t in time:
    #genera una barra de avance porcentual del programa
    if (t*100/tiempo_total)%5 < .3: print(f"{int(t*100/tiempo_total)} %")
    
    for i in particulas:
        # Colisiones con los límites horizontales
        #if i.r[0] - i.radio <= -limx:
        #    i.r[0] = -limx + i.radio  # Reposicionar justo en el borde
        #    i.v[0] = -i.v[0]  # Invertir velocidad
        #elif i.r[0] + i.radio >= limx:
        #    i.r[0] = limx - i.radio  # Reposicionar justo en el borde
        #    i.v[0] = -i.v[0]  # Invertir velocidad

        # Colisiones con los límites verticales
        #if i.r[1] - i.radio <= -limy:
        #    i.r[1] = -limy + i.radio  # Reposicionar justo en el borde
        #    i.v[1] = -i.v[1]  # Invertir velocidad
        #elif i.r[1] + i.radio >= limy:
        #    i.r[1] = limy - i.radio  # Reposicionar justo en el borde
        #    i.v[1] = -i.v[1]  # Invertir velocidad

        # Frontera periódica en x
        if i.r[0] - i.radio < -limx:
            i.r[0] = limx - i.radio  # Reaparece en el lado derecho
            contador += 1
        elif i.r[0] + i.radio > limx:
            i.r[0] = -limx + i.radio  # Reaparece en el lado izquierdo
        # Frontera periódica en y
        if i.r[1] - i.radio < -limy:
            i.r[1] = limy - i.radio  # Reaparece en el lado derecho
        elif i.r[1] + i.radio > limy:
            i.r[1] = -limy + i.radio  # Reaparece en el lado izquierdo
        # Frontera periódica en z
        if i.r[2] - i.radio < -limz:
            i.r[2] = limz - i.radio  # Reaparece en el lado derecho
        elif i.r[2] + i.radio > limz:
            i.r[2] = -limz + i.radio  # Reaparece en el lado izquierdo

        
        
    # Reemplazar el ciclo de colisión por la versión optimizada
    for i, p1 in enumerate(particulas):
        for p2 in particulas[i+1:]:
            p1.colision(p2,t)
    
    # Movimiento de las partículas
    for i in particulas:
        i.campo_electrico(E,dt)
        i.mov(dt)



print(f"En {tiempo_total}s pasaron {contador} electrones")

# SE VAN A PROMEDIAR TODOS LOS TAU DE CADA PARTICULA
prom_tau=[]
for i in particulas:
    if i.movimiento==True and len(i.tau)>1:
        prom=prom_tau.append(np.mean(i.tau[1:]))
print(f"El tiempo de deriva promedio tau fue {np.mean(prom_tau)} segundos")

# Calcular las coordenadas x e y de la partícula en función del tiempo
#Lista de posiciones
x,y,z,T,radios,colores=[],[],[],[],[],[]

for i in particulas:
    x.append(np.array(i.px))
    y.append(np.array(i.py))
    z.append(np.array(i.pz))
    T.append(time)
    #Se crea una lista de radios para cada particula del tamaño del vector posiciones
    radios.append([i.radio] * len(i.px))
    colores.append([i.color] * len(i.px))


#Concatenar las listas
X=np.concatenate(x)
Y=np.concatenate(y)
Z=np.concatenate(z)
TI=np.concatenate(T)
R=np.concatenate(radios)
C=np.concatenate(colores)
