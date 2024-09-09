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
        self.r = np.array([x, y])
        self.v = np.array([vx, vy])
        self.radio=radio
        self.m = m
        self.px = []
        self.py = []
        self.q=carga
        self.color=color
        self.movimiento=movimiento
    
    def mov(self, dt):
        self.px.append(self.r[0])
        self.py.append(self.r[1])
        if self.movimiento==True:
            self.r=self.r+self.v*dt
        
    def colision(self, particula2=None):        
        # Verifica si las particulas se tocan:
        D2 = (particula2.r[0] - self.r[0])*(particula2.r[0] - self.r[0]) + (particula2.r[1] - self.r[1])*(particula2.r[1] - self.r[1])
        suma_radios = self.radio + particula2.radio
        # separacion entre particulas:
        d=np.array(particula2.r)-np.array(self.r)
        # Velocidad relativa entre particulas:
        vrel=particula2.v-self.v
        if self!=particula2 and D2 <= suma_radios*suma_radios and np.dot(vrel,d)<0:
            self.v=-self.v 
            particula2.v=-particula2.v
            # Cálculo de las velocidades después de la colisión en una dimensión
            #v1 = self.v
            #v2= particula2.v
            #self.v = (v1*(self.m-particula2.m)+2*v2 * particula2.m / (self.m + particula2.m)) 
            #particula2.v = (v2*(particula2.m-self.m)+2*v1 * self.m / (self.m + particula2.m))  


    def campo_electrico(self,E,dt):
        a=self.q*E/self.m
        self.v=a*dt+self.v
        
        
        



# Definimos un campo electrico en dirección x:
E=np.array([1*10**(-10),0.])




# CREACIÓN DEL DATA FRAME
# Definir los parámetros del movimiento

tiempo_total = 5  # Tiempo total del movimiento (segundos)
num_puntos = 1000  # Número de puntos en el dataframe  1000
dt=tiempo_total/num_puntos
time = np.linspace(0, tiempo_total,num_puntos)

# LIMITES DE LA CAJA
limx=.9
limy=.9
# NUMERO DE IONES MXM
M=3
M2=M**2
# NUMERO DE ELECTRONES
N=M2*4
#Lista de particulas:
particulas=[]
#Definir radio de electrones, protones y distancia entre protones en la red:
r_e, m_e=.03, constants.m_e
r_p, m_p=.05, constants.m_p

# Definir cuadricula de iones, o atomos y agregarlos a la lista de particula
for i in range(M):
    for j in range(M):
        particulas.append(particula((-limx+r_p*1.1)+i/2,(-limy+r_p*1.1)+j/2,0,0,r_p,m_p,color="red",movimiento=False))
# Definir electrones y agregarlos a la lista particulas, se tiene que tener en cuenta que no se pueden solapar entre ellos ni con los radios de los protones
# Entonces si se genera alguno que se solape con los demas, se genera otra posicion 
for i in range(N):
    gx, gy=random.uniform(-limx+r_e,limx-r_e),random.uniform(-limy+r_e,limy-r_e)
    # Generar nuevas posiciones gx y gy aleatoriamente hasta que no solapen con ninguna partícula
    for i in particulas:
        solapado = True
        while solapado:
            gx = random.uniform(-limx + r_e, limx - r_e)
            gy = random.uniform(-limy + r_e, limy - r_e)
            # Verificar si la nueva posición (gx, gy) no solapa con ninguna partícula
            solapado = False
            for p in particulas:
                if p != i:  # No verificar contra sí misma
                    if ((gx-p.r[0])**2+(gy-p.r[1])**2)**0.5 < (r_e + p.radio):
                        solapado = True
                        break
    gvx, gvy=random.uniform(-limx,limx),random.uniform(-limy,limy)*20
    #gvx, gvy=0,0
    particulas.append(particula(gx,gy,gvx,gvy,r_e,m_e,color="purple"))


contador=0
for t in time:
    for i in particulas:
        # Colisiones con los límites horizontales
        #if i.r[0] - i.radio <= -limx:
        #    i.r[0] = -limx + i.radio  # Reposicionar justo en el borde
        #    i.v[0] = -i.v[0]  # Invertir velocidad
        #elif i.r[0] + i.radio >= limx:
        #    i.r[0] = limx - i.radio  # Reposicionar justo en el borde
        #    i.v[0] = -i.v[0]  # Invertir velocidad

        # Colisiones con los límites verticales
        if i.r[1] - i.radio <= -limy:
            i.r[1] = -limy + i.radio  # Reposicionar justo en el borde
            i.v[1] = -i.v[1]  # Invertir velocidad
        elif i.r[1] + i.radio >= limy:
            i.r[1] = limy - i.radio  # Reposicionar justo en el borde
            i.v[1] = -i.v[1]  # Invertir velocidad

        # Frontera periodica en x
        if i.r[0] - i.radio <= -limx:
            gx = limx-i.radio
            gy = random.uniform(-limx + r_e, limx - r_e)
            gvx, gvy=random.uniform(-limx,limx),random.uniform(-limy,limy)
            i.r=[gx, gy]  # Reaparece en el lado derecho
            i.v[0]=gvx
            i.v[1]=gvy
            contador=contador+1
        #elif i.r[0] + i.radio >= limx:
        #    i.r[0] = -limx + i.radio  # Reaparece en el lado izquierdo
        
    # Reemplazar el ciclo de colisión por la versión optimizada
    for i, p1 in enumerate(particulas):
        for p2 in particulas[i+1:]:
            p1.colision(p2)
    
    # Movimiento de las partículas
    for i in particulas:
        i.campo_electrico(E,dt)
        i.mov(dt)
print(f"En {tiempo_total}s pasaron {contador} electrones")

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
#plt.show()

# Guardar la animación
ani.save('animacion_DRUDE_V2.mp4', writer='ffmpeg')