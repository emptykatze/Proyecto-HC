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
#Definir clase particula
class particula:
    def __init__(self, x, y, vx, vy, radio=100,m=1,carga=-constants.elementary_charge,movimiento=True,color="purple"):
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
        self.q=carga
        self.color=color
        self.movimiento=movimiento

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
        if self.movimiento==True:
            self.r = self.r + self.v * dt
            # Se actualizan las componentes individuales de x e y
            self.x,self.y=self.r
        
    def colision(self,particula2=None):
        # Verifica si la distancia entre las particulas es menor a la suma de sus radios
        #distancia entre las particulas
        D2=(particula2.x-self.x)*(particula2.x-self.x)+(particula2.y-self.y)*(particula2.y-self.y)
        if self!=particula2 and D2<=(particula2.radio+self.radio)*(particula2.radio+self.radio):
            a=self.v
            self.v=(a*(self.m-particula2.m)+2*particula2.m*particula2.v)/(self.m+particula2.m)
            particula2.v=(particula2.v*(particula2.m-self.m)+2*self.m*a)/(self.m+particula2.m)
            # Cambia las posiciones para que las particulas no se sobrepongan
            #Calcula la superposicion y direccion de superposicion
            superposicion = (particula2.radio + self.radio) - np.sqrt(D2)
            direccion = (self.r - particula2.r) / np.sqrt(D2)
            # Actualiza las posiciones
            if self. movimiento==True:
                self.r += direccion * superposicion / 2
            if particula2. movimiento==True:
                particula2.r -= direccion * superposicion / 2
    def campo_electrico(self,E,dt):
        a=self.q*E/self.m
        self.v=a*dt+self.v
        self.vx,self.vy=self.v
        
        
        



# Definimos un campo electrico en dirección x:
E=np.array([10**(-10)/8,0.])




# CREACIÓN DEL DATA FRAME
# Definir los parámetros del movimiento

tiempo_total = 20  # Tiempo total del movimiento (segundos)
num_puntos = 1000  # Número de puntos en el dataframe  1000
dt=tiempo_total/num_puntos
time = np.linspace(0, tiempo_total,num_puntos)

#movimiento
limx=.9
limy=.9
#Particulas:
N=50
#Lista de particulas:
particulas=[]
#Definir radio de electrones, protones y distancia entre protones en la red:
r_e=.03
m_e=constants.m_e
r_p=.05
m_p=constants.m_p
d_p=limx/3
# Definir cuadricula de protones, o atomos y agregarlos a la lista de particula
for i in range(10):
    for j in range(10):
        particulas.append(particula((-limx+r_p*1.1)+i/5,(-limy+r_p*1.1)+j/5,0,0,r_p,m_p,color="red",movimiento=False))

# Definir electrones y agregarlos a la lista particulas, se tiene que tener en cuenta que no se pueden solapar entre ellos ni con los radios de los protones
# Entonces si se genera alguno que se solape con los demas, se genera otra posicion 
for i in range(N):
    gx, gy=random.uniform(-limx+r_e,limx-r_e),random.uniform(-limy+r_e,limy-r_e)
    gvx, gvy=random.uniform(-limx,limx)/20,random.uniform(-limy,limy)/20
    particulas.append(particula(gx,gy,gvx,gvy,r_e,m_e,color="purple"))



for t in time:
    for i in particulas:
        #if i.r[0]-i.radio <= -limx or i.r[0]+i.radio >= limx:
        #    i.setvx(-i.v[0])
        #Condicion de frontera periodica, las particulas que salen por izquierda entran por derecha
        if i.r[0]-i.radio <= -limx:
            i.pos(limx-i.radio,i.y)
            i.vel(random.uniform(-limx,limx),random.uniform(-limy,limy))
        if i.r[1]-i.radio <= -limy or i.r[1]+i.radio >= limy:
            i.setvy(-i.v[1])
        for j in particulas:
            i.colision(j)
        i.campo_electrico(E,dt)
        i.mov(dt)   
    

# Calcular las coordenadas x e y de la partícula en función del tiempo
#Lista de posiciones
x,y,T,radios,colores=[],[],[],[],[]

for i in particulas:
    x.append(np.array(i.px))
    y.append(np.array(i.py))
    T.append(time)
    #Se crea una lista de radios para cada particula del tamaño del vector posiciones
    radios.append([i.radio] * len(i.px))
    colores.append([i.color] * len(i.px))


#Concatenar las listas
X=np.concatenate(x)
Y=np.concatenate(y)
TI=np.concatenate(T)
R=np.concatenate(radios)
C=np.concatenate(colores)

# Crear el DataFrame con las coordenadas x, y y radios
df = pd.DataFrame({"time": TI, "X": X, "Y": Y, "radios": R,"colores":C})

# Crear una figura y ejes
fig, ax = plt.subplots()

# Definir los límites de los ejes
ax.set_xlim(-limx, limx)
ax.set_ylim(-limy, limy)

# Crear una colección de elipses vacía, se utilizó como referencia https://stackoverflow.com/questions/33094509/correct-sizing-of-markers-in-scatter-plot-to-a-radius-r-in-matplotlib
# Utilizamos elipses ya que es mas sencillo decirle a matplot cuan debe ser el diametro que se ve en la grafica 
patches = []
collection = PatchCollection(patches, alpha=0.6)
ax.add_collection(collection)
ax.grid(True)

# Función de inicialización para la animación
def init():
    collection.set_paths([])
    return collection,

# Función de actualización para la animación
def update(frame):
    data = df[df['time'] == frame]
    x_data = data['X'].values
    y_data = data['Y'].values
    sizes = data['radios'].values   
    colores=data["colores"]
    # Crear nuevas elipses para el frame actual
    patches = []
    for (x, y, r,c) in zip(x_data, y_data, sizes,colores):
        ellipse = Ellipse(xy=(x, y), width=2*r, height=2*r, angle=0, edgecolor='none', facecolor=c)
        patches.append(ellipse)
    
    
    # Actualizar la colección de parches
    collection.set_paths(patches)
    collection.set_facecolor(colores)
    return collection,

# Crear la animación
ani = animation.FuncAnimation(
    fig, update, frames=np.unique(df['time']),
    init_func=init, repeat=False,
    interval=15
)


fig.tight_layout()
plt.show()