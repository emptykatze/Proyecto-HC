import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import random
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from scipy import constants
from scipy.stats import maxwell
from scipy.stats import norm

"""
Se utilizaron unidades de picometros para la distancia, para que la distancia entre iones de la red no
tuvierta un numero muy grande, por lo que se utilizó 1 m = 1e12 pm
"""
k_b=constants.Boltzmann #Constante de boltzmann en (Kg pm^2/ps^2)/K
T=77.
m_e=constants.electron_mass
velocidad_cuadratica_promedio=np.sqrt(2*k_b * T / m_e)
def random_vel2(t=0,vd=0):
        if t==0: return [np.random.normal(vd, velocidad_cuadratica_promedio),np.random.normal(0,velocidad_cuadratica_promedio)]
        if t!=0: return velocidad_cuadratica_promedio
def random_vel(t=0,vd=0):
        angulo=random.uniform(0,2*np.pi)
        magnitud_vel=maxwell.rvs(scale=velocidad_cuadratica_promedio)
        if t==0: return [magnitud_vel*np.sin(angulo),magnitud_vel*np.cos(angulo)]
        if t!=0: return velocidad_cuadratica_promedio
#Definir clase particula
class particula:
    def __init__(self, x, y, vx, vy, radio=100,radio_animacion=100,m=1,carga=-constants.elementary_charge,movimiento=True,color="purple"):
        self.r = np.array([x, y])
        self.v = np.array([vx, vy])
        self.radio=radio
        self.radio_animacion=radio_animacion
        self.m = m
        self.px = []
        self.py = []
        self.vx = []
        self.vy = []
        self.v2 = []
        self.v2raiz = []
        self.q=carga
        self.color=color
        self.movimiento=movimiento
        self.tau=[0]
    
    def mov(self, dt):
        self.px.append(self.r[0])
        self.py.append(self.r[1])
        self.vx.append(self.v[0])
        self.vy.append(self.v[1])
        self.v2.append(self.v[0]*self.v[0]+self.v[1]*self.v[1])
        self.v2raiz.append((self.v[0]*self.v[0]+self.v[1]*self.v[1])*.5)

        if self.movimiento==True:
            self.r=self.r+self.v*dt
        
    def colision(self, particula2=None,t=0,vd=0):        
        # Verifica si las particulas se tocan:
        D2 = (particula2.r[0] - self.r[0])*(particula2.r[0] - self.r[0]) + (particula2.r[1] - self.r[1])*(particula2.r[1] - self.r[1])
        suma_radios = self.radio + particula2.radio
        # separacion entre particulas:
        d=np.array(particula2.r)-np.array(self.r)
        # Velocidad relativa entre particulas:
        vrel=particula2.v-self.v
        if self!=particula2 and D2 <= suma_radios*suma_radios and np.dot(vrel,d)<0 and (particula2.movimiento==False or self.movimiento==False):
            
            normal = d / np.linalg.norm(d)
            overlap = suma_radios - np.sqrt(D2)
            correction = normal * overlap / 2

            if particula2.movimiento==False: 
                nueva_velocidad=random_vel2(0,vd)
                self.v[0]=nueva_velocidad[0]
                self.v[1]=nueva_velocidad[1]
                self.r -= correction
                self.v=-self.v
                self.tau.append(t-self.tau[-1])

            else: 
                nueva_velocidad=random_vel2(0,vd)
                particula2.v[0]=nueva_velocidad[0]
                particula2.v[1]=nueva_velocidad[1]
                particula2.r += correction
                particula2.v=-particula2.v 
                particula2.tau.append(t-particula2.tau[-1])
            
            
            
            
            # Si hay un choque se va a guardar el tiempo en el que ocurre menos el tiempo anterior
            
    def campo_electrico(self,E,dt):
        a=self.q*E/self.m
        self.v=a*dt+self.v

# Definimos un campo electrico en dirección x:

E=np.array([10**-4,0.])

#Distacia de la red en pm
d=409

# NUMERO DE IONES MXM
M=6
M2=M**2
# NUMERO DE ELECTRONES
N=M2

#Lista de particulas:
particulas=[]
#Definir radio de electrones, protones y distancia entre protones en la red:
r_e, m_e=d/16, constants.electron_mass
r_p, m_p=d/4, constants.m_p

# LIMITES DE LA CAJA
limx=((M)*d+2.1*r_p)/2
limy=((M)*d+2.1*r_p)/2
# Definir cuadricula de iones, o atomos y agregarlos a la lista de particula
for i in range(M):
    for j in range(M):
        particulas.append(particula((-limx+r_p*1.1)+i*d+d/2,(-limy+r_p*1.1)+j*d+d/2,0,0,r_p,r_p,m_p,color="red",movimiento=False))
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
    velocidad_aleatoria=random_vel2()
    gvx=velocidad_aleatoria[0]
    gvy=velocidad_aleatoria[1]
    
    #gvx, gvy=0,0
    particulas.append(particula(gx,gy,gvx,gvy,r_e,r_e,m_e,color="purple"))


#TIEMPO REAL DE SIMULACION
# El dt es dos ordenes de magnitud mas pequeño que el tiempo en el que una particula atravieza todo el sistema

dt=(2*limx/random_vel2(2))/200 #  0.00028310447443648343

tiempo_total =dt*20000  # Tiempo total del movimiento (pico segundos)

num_puntos = int(tiempo_total/dt)  # Número de puntos en el dataframe  10 000
#dt=tiempo_total/num_puntos
time = np.linspace(0, tiempo_total,num_puntos)
#Cuenta los electrones que pasan por el lado izquierdo de la simulacion
contador=0
vd=0
for t in time:
    velocidad_promedio_en_x=[]
    #genera una barra de avance porcentual del programadld¿90                    
    if (t*100/tiempo_total)%1 < .3 and int(t*100/tiempo_total)!=int((t-dt)*100/tiempo_total): print(f"{int(t*100/tiempo_total)} %")
    
    for i in particulas:
        if i.movimiento:
            # Frontera periódica en x
            if i.r[0] - i.radio < -limx:
                i.r[0] = 2*limx - i.radio+ i.r[0]  # Reaparece en el lado derecho
                contador += 1
            elif i.r[0] + i.radio > limx:
                i.r[0] = -2*limx + i.radio-i.r[0]  # Reaparece en el lado izquierdo
            # Frontera periódica en y
            if i.r[1] - i.radio < -limy:
                i.r[1] = 2*limx - i.radio +i.r[1]  # Reaparece en el lado derecho
            elif i.r[1] + i.radio > limy:
                i.r[1] = -2*limy + i.radio- i.r[1]  # Reaparece en el lado izquierdo
        # VELOCIDAD PROMEDIO EN X
        if i.movimiento==True and len(i.vx)>0:
            velocidad_promedio_en_x.append(np.mean(i.vx))
    if len(velocidad_promedio_en_x)>0:
        vd=np.mean(velocidad_promedio_en_x)
        
    # Reemplazar el ciclo de colisión por la versión optimizada
    for i, p1 in enumerate(particulas):
        for p2 in particulas[i+1:]:
            p1.colision(p2,t,vd)
    
        
    # Movimiento de las partículas
    for i in particulas:
        i.campo_electrico(E,dt)
        i.mov(dt)
        
with open('vx.dat', 'w') as file:
    for i in particulas:
        # Escribir cada fila en el archivo separando los valores por un espacio
        if i.movimiento:    file.write(' '.join(map(str, i.vx)) + '\n')
with open('vy.dat', 'w') as file:
    for i in particulas:
        # Escribir cada fila en el archivo separando los valores por un espacio
        if i.movimiento:    file.write(' '.join(map(str, i.vy)) + '\n')

# Abrir el archivo de velocidades 
file = open('velocidades.dat', 'w')
# Escribir datos en el archivo de velocidades
for i in particulas:
    if i.movimiento:    file.write(' '.join(map(str, i.vx)) )
file.write("\n")
for i in particulas:
    if i.movimiento:    file.write(' '.join(map(str, i.vy)) )
# Cerrar el archivo 
file.close() 

#Escribir datos en el archivo para los tiempos de colisión 
file = open('tau.dat', 'w')
for i in particulas:
    if i.movimiento:    file.write(' '.join(map(str, i.tau[1:] ))+"\n" )
file.close()

#file = open('velocidades_sergio2.dat', 'w')
#PROMEDIO DE VELOCIDADES Y VELOCIDADES CUADRATICAS
#promvx,promvy=np.mean(prom_vx),np.mean(prom_vy)
#velprom=(promvx*promvx+promvy*promvy)**.5
#promv2=np.sqrt(np.mean(prom_v2))





# HISTOGRAMA PARA X al final
#plt.figure(figsize=(10, 5))
#plt.title('Histograma velocidades en X para el final')
    #Agrega una linea vertical en la velocidad de deriva
#plt.axvline(x=promvx, color='darkblue', linestyle='--', linewidth=3, label='Velocidad de deriva')
#VX= np.linspace( promvx- 4*velocidad_cuadratica_promedio, promvx+4*velocidad_cuadratica_promedio, 1000)
#DATAVX= norm.pdf(VX, promvx, velocidad_cuadratica_promedio)
#f.write(str(DATAVX)+"\n")
#plt.plot(VX, DATAVX, 'darkblue', linewidth=2)
#plt.hist(histogramaxfinal, bins=50, density=True,color="royalblue")
#plt.legend()
#plt.savefig("Histograma_Vel_X_TODO_T")
#f.close()


# HISTOGRAMA PARA LA VELOCIDAD CUADRATICA
#plt.figure(figsize=(10, 5))
#plt.title('Histograma Velocidad cuadratica')
    #Agrega una linea vertical en la velocidad de deriva
#plt.axvline(x=promvx, color='darkblue', linestyle='--', linewidth=3, label='Velocidad de deriva')
#VX= np.linspace( 0, promvx+4*velocidad_cuadratica_promedio, 1000)
#DATAVX= ((norm.pdf(VX, promvx, velocidad_cuadratica_promedio)**2)+(norm.pdf(VX, 0, velocidad_cuadratica_promedio)**2))**.5
#plt.plot(VX, DATAVX, 'darkblue', linewidth=2)
#plt.hist(histogramav2raiz, bins=50, density=True,color="royalblue")
#plt.legend()
#plt.savefig("Histograma_velocidad_cuadratica")

#Densidad de electrones
densidad_e=(N/(2*limx*2*limy))*10**-24
#campo electrico en voltios / metro:

# PONER DATOS DE LA SIMULACION EN UN RESULTADOS.DAT
# Encabezados en una lista
encabezados = [
    f"Tiempo_total[ps]{tiempo_total/dt}dt",
    "dt[ps]",
    "Contador[electrones]",
    "Tamano_x[pm]",
    "Tamano_y[pm]",
    "E_x[V/pm]",
    "E_y[V/pm]",
    "Num_iones",
    "Num_electrones"
]

# Empaquetar los datos en un array (asegurarse de que 'E' sea un array o lista si tiene más de una componente)
datos = np.array([
    tiempo_total,
    dt,
    contador,            # Electrones que pasaron en la dirección del campo
    2 * limx,            # tamaño total en x
    2 * limy,            # tamaño total en y
    E[0],                # componente x del campo eléctrico
    E[1],                # componente y del campo eléctrico
    M2,                  # número de iones
    N                    # número de electrones
])

# Crear una lista con filas donde cada fila tiene el encabezado y su valor correspondiente
# Usamos zip para emparejar encabezados y datos
data_with_headers = np.column_stack((encabezados, datos))
# Guardar el archivo .dat con un formato apropiado
np.savetxt('resultados.dat', data_with_headers, fmt='%s', delimiter="   ")


#########################################
### LISTAS PARA INICIAR LA ANIMACIÓN  ###
#########################################
x,y,T,radios,colores=[],[],[],[],[]

for i in particulas:
    x.append(np.array(i.px))
    y.append(np.array(i.py))
    T.append(time)
    #Se crea una lista de radios para cada particula del tamaño del vector posiciones
    radios.append([i.radio_animacion] * len(i.px))
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
# Texto para el temporizador
texto_tiempo= ax.text(0.02, 0.95, '', transform=ax.transAxes)
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
        ellipse = Ellipse(xy=(x, y), width=2*r, height=2*r, angle=0, edgecolor="none", facecolor=c)
        patches.append(ellipse)
    
    
    # Actualizar la colección de parches
    collection.set_paths(patches)
    collection.set_facecolor(colores)
     # Actualizar el temporizador
    tiempo_actual = frame * dt/(4*10**-16)
    texto_tiempo.set_text(f"Tiempo: {tiempo_actual:.2e} s")
    
    # Devolver la colección y el texto actualizado
    return collection, tiempo_actual

# Crear la animación
ani = animation.FuncAnimation(
    fig, update, frames=np.unique(df['time']),
    init_func=init, repeat=False,
    interval=15)


fig.tight_layout()
# Guardar la animación
#ani.save('animacion_DRUDE_V5.mp4', writer='ffmpeg')

#plt.show()