import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import random
#Definir clase particula
class particula:
    def __init__(self, x, y, vx, vy, radio=5,m=1):
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
    
    def setvy(self, nvy):
        self.v[1] = nvy
    
    def mov(self, dt):
        self.px.append(self.r[0])
        self.py.append(self.r[1])
        self.r = self.r + self.v * dt




# CREACIÓN DEL DATA FRAME
# Definir los parámetros del movimiento

tiempo_total = 20  # Tiempo total del movimiento (segundos)
num_puntos = 1000  # Número de puntos en el dataframe
dt=tiempo_total/num_puntos
time = np.linspace(0, tiempo_total,num_puntos)

#movimiento
limx=10
limy=10
#Particulas:
N=500

particulas=[]
#Lista de particulas:
for i in range(N):
    particulas.append(particula(random.randint(-limx,limx),random.randint(-limy,limy),random.randint(-limx,limx),random.randint(-limy,limy)))

for t in time:
    for i in particulas:
        if np.abs(i.r[0]) > limx:
            i.setvx(-i.v[0])
        if np.abs(i.r[1]) > limy:
            i.setvy(-i.v[1])
        i.mov(dt)
    

# Calcular las coordenadas x e y de la partícula en función del tiempo
#Lista de posiciones
x,y,T=[],[],[]

for i in particulas:
    x.append(np.array(i.px))
    y.append(np.array(i.py))
    T.append(time)

#Concatenar las listas
X=np.concatenate(x)
Y=np.concatenate(y)
TI=np.concatenate(T)
# Crear el dataframe con las coordenadas x e y

df = pd.DataFrame({'time': TI, 'X': X, 'Y': Y})


#df = px.data.gapminder()


fig = px.scatter(
    data_frame=df,
    x='X',
    y='Y', 
    animation_frame='time', 
    range_x=[-limx,limx], 
    range_y=[-limy,limy], 
    color_discrete_sequence=['red'],
)

#Hacer que los ejes de la figura no se deformen
fig.update_layout(
    xaxis=dict(scaleanchor="y", scaleratio=1),
    yaxis=dict(scaleanchor="x", scaleratio=1)
)
#vel
# Ajustar la velocidad de la animación
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 5  # Duración en milisegundos por frame
#limites caja
fig.add_vline(x=limx, line_color="red")
fig.add_vline(x=-limx, line_color="red")
fig.add_hline(y=limy, line_color="red")
fig.add_hline(y=-limy, line_color="red")


fig.show()
