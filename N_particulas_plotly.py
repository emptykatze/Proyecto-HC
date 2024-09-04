import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import random
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
#Definir clase particula
class particula:
    def __init__(self, x, y, vx, vy, radio=100,m=1):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.r = np.array([x, y])
        self.v = np.array([vx, vy])
        self.radio=radio
        self.m = m
        self.px = []
        self.py = []

    # definir funciones de particula
    def pos(self, nuevox, nuevoy):
        self.r = np.array([nuevox, nuevoy])
    
    def vel(self, nuevovx, nuevovy):
        self.v = np.array([nuevovx, nuevovy])
    
    def setvx(self, nvx):
        self.v[0] = nvx
    
    def getx(self):
        return self.x
    def setvy(self, nvy):
        self.v[1] = nvy
    
    def mov(self, dt):
        self.px.append(self.r[0])
        self.py.append(self.r[1])
        self.r = self.r + self.v * dt
    #def choque(self,nvx,nvy):





# CREACIÓN DEL DATA FRAME
# Definir los parámetros del movimiento

tiempo_total = 20  # Tiempo total del movimiento (segundos)
num_puntos = 1000  # Número de puntos en el dataframe
dt=tiempo_total/num_puntos
time = np.linspace(0, tiempo_total,num_puntos)

#movimiento
limx=1
limy=1
#Particulas:
N=10

particulas=[]
#Lista de particulas:
r=.05
for i in range(N):
    particulas.append(particula(random.uniform(-limx+r,limx-r),random.uniform(-limy+r,limy-r),random.uniform(-limx,limx),random.uniform(-limy,limy),r))
#particulas=[particula(0,0,0,0,r)]
for t in time:
    for i in particulas:
        if i.r[0]-i.radio <= -limx or i.r[0]+i.radio >= limx:
            i.setvx(-i.v[0])
        if i.r[1]-i.radio <= -limy or i.r[1]+i.radio >= limy:
            i.setvy(-i.v[1])
        for j in particulas:
            if i!=j:
                r_ij=np.sqrt((i.x-j.x)*(i.x-j.x)+(i.y-j.y)*(i.y-j.y))
        i.mov(dt)
    

# Calcular las coordenadas x e y de la partícula en función del tiempo
#Lista de posiciones
x,y,T,radios=[],[],[],[]

for i in particulas:
    x.append(np.array(i.px))
    y.append(np.array(i.py))
    T.append(time)
    #Se crea una lista de radios para cada particula del tamaño del vector posiciones
    radios.append([i.radio] * len(i.px))


#Concatenar las listas
X=np.concatenate(x)
Y=np.concatenate(y)
TI=np.concatenate(T)
R=np.concatenate(radios)

# Crear el DataFrame con las coordenadas x, y y radios
df = pd.DataFrame({'time': TI, 'X': X, 'Y': Y, 'radios': R})

# Crear una figura y ejes
fig, ax = plt.subplots()

# Definir los límites de los ejes
ax.set_xlim(-limx, limx)
ax.set_ylim(-limy, limy)

# Crear una colección de elipses vacía
patches = []
collection = PatchCollection(patches, facecolor='purple', alpha=0.6)
ax.add_collection(collection)
ax.grid(True)

# Función de inicialización para la animación
def init():
    collection.set_paths([])
    return collection,

# Función de actualización para la animación
def update(frame):
    # Filtrar datos para el frame actual
    data = df[df['time'] == frame]
    x_data = data['X'].values
    y_data = data['Y'].values
    sizes = data['radios'].values   # Ajustar el factor de escala del tamaño si es necesario

    # Crear nuevas elipses para el frame actual
    patches = []
    for (x, y, r) in zip(x_data, y_data, sizes):
        ellipse = Ellipse(xy=(x, y), width=2*r, height=2*r, angle=0, edgecolor='none', facecolor='purple')
        patches.append(ellipse)
    
    collection.set_paths(patches)
    return collection,

# Crear la animación
ani = animation.FuncAnimation(
    fig, update, frames=np.unique(df['time']),
    init_func=init, repeat=False,
    interval=15
)

# Ajustar el tamaño del gráfico al tamaño del área de dibujo
fig.tight_layout()

# Mostrar la animación
plt.show()