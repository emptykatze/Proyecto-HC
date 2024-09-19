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
T=273.15+20
m_e=constants.electron_mass
velocidad_cuadratica_promedio=np.sqrt(2*k_b * T / m_e)
def random_vel2(t=0,vd=0):
        if t==0: return [np.random.normal(vd, velocidad_cuadratica_promedio),np.random.normal(vd, velocidad_cuadratica_promedio)]
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
            else: 
                nueva_velocidad=random_vel2(0,vd)
                particula2.v[0]=nueva_velocidad[0]
                particula2.v[1]=nueva_velocidad[1]
                particula2.r += correction
                particula2.v=-particula2.v  
            
        # Tiempo entre choques
        #Para particula2
            tau2=t-particula2.tau[-1]
            tau1=t-self.tau[-1]
            
            self.tau.append(tau1)
            particula2.tau.append(tau2)
            
            
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
    particulas.append(particula(gx,gy,gvx,gvy,r_e*0,r_e,m_e,color="purple"))


#TIEMPO REAL DE SIMULACION
# El dt es dos ordenes de magnitud mas pequeño que el tiempo en el que una particula atravieza todo el sistema

dt=(2*limx/random_vel2(2))/100 #  0.00028310447443648343

tiempo_total =dt*10000*3  # Tiempo total del movimiento (segundos)

num_puntos = int(tiempo_total/dt)  # Número de puntos en el dataframe  10 000
#dt=tiempo_total/num_puntos
time = np.linspace(0, tiempo_total,num_puntos)
#Cuenta los electrones que pasan por el lado izquierdo de la simulacion
contador=0
vd=0
for t in time:
    velocidad_promedio_en_x=[]
    #genera una barra de avance porcentual del programa
    if (t*100/tiempo_total)%5 < .3 and int(t*100/tiempo_total)!=int((t-dt)*100/tiempo_total): print(f"{int(t*100/tiempo_total)} %")
    
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
        
        if i.movimiento:
            # Frontera periódica en x
            if i.r[0] - i.radio < -limx:
                i.r[0] = 2*limx - i.radio+ i.r[0]  # Reaparece en el lado derecho
                contador += 1
            elif i.r[0] + i.radio > limx:
                i.r[0] = -limx + i.radio-i.r[0]  # Reaparece en el lado izquierdo
            # Frontera periódica en y
            if i.r[1] - i.radio < -limy:
                i.r[1] = limx - i.radio +i.r[1]  # Reaparece en el lado derecho
            elif i.r[1] + i.radio > limy:
                i.r[1] = -limy + i.radio- i.r[1]  # Reaparece en el lado izquierdo
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
        




print(f"En {tiempo_total}s pasaron {contador} electrones")

# SE VAN A PROMEDIAR TODOS LOS TAU DE CADA PARTICULA
prom_tau=[]
prom_vx,prom_vy=[],[]
prom_v2=[]

histogramay=[]
histogramax=[]
histogramaxfinal=[]
for i in particulas:
    if i.movimiento==True and len(i.tau)>1:
        prom_tau.append(np.mean(i.tau[1:]))
        prom_vx.append(np.mean((i.vx)))
        prom_vy.append(np.mean((i.vy)))
        prom_v2.append(np.mean(i.v2))
        histogramay+=i.vy
        histogramax+=i.vx
        histogramaxfinal+=i.vx[-15:-1]

#PROMEDIO DE VELOCIDADES Y VELOCIDADES CUADRATICAS
promvx,promvy=np.mean(prom_vx),np.mean(prom_vy)
velprom=(promvx*promvx+promvy*promvy)**.5
promv2=np.sqrt(np.mean(prom_v2))

#HISTOGRAMA PARA Y
plt.figure(figsize=(10, 5))
plt.title('Histograma velocidades en Y')
VY= np.linspace( - 4*velocidad_cuadratica_promedio,   4*velocidad_cuadratica_promedio, 1000)
DATAVY= norm.pdf(VY, 0, velocidad_cuadratica_promedio)
plt.plot(VY, DATAVY, color='darkslateblue', linewidth=3)
plt.hist(histogramay, bins=60, density=True,color="mediumslateblue")
plt.savefig("Histograma_Vel_Y")
# HISTOGRAMA PARA X EN TODOS LOS TIEMPOS
plt.figure(figsize=(10, 5))
plt.title('Histograma velocidades en X para todos los tiempos')
    #Agrega una linea vertical en la velocidad de deriva
plt.axvline(x=promvx, color='yellowgreen', linestyle='--', linewidth=3, label='Velocidad de deriva')
VX= np.linspace( promvx- 4*velocidad_cuadratica_promedio, promvx+4*velocidad_cuadratica_promedio, 1000)
DATAVX= norm.pdf(VX, promvx, velocidad_cuadratica_promedio)
plt.plot(VX, DATAVX, 'yellowgreen', linewidth=2)
plt.hist(histogramax, bins=60, density=True,color="forestgreen")
plt.legend()
plt.savefig("Histograma_Vel_X_TODO_T")


# HISTOGRAMA PARA X al final
plt.figure(figsize=(10, 5))
plt.title('Histograma velocidades en X para el final')
    #Agrega una linea vertical en la velocidad de deriva
plt.axvline(x=promvx, color='darkblue', linestyle='--', linewidth=3, label='Velocidad de deriva')
VX= np.linspace( min(histogramaxfinal), max(histogramaxfinal), 1000)
DATAVX= norm.pdf(VX, promvx, velocidad_cuadratica_promedio)
plt.plot(VX, DATAVX, 'darkblue', linewidth=2)
plt.hist(histogramaxfinal, bins=60, density=True,color="royalblue")
plt.legend()
plt.savefig("Histograma_Vel_X_TODO_T")

#Densidad de electrones
densidad_e=N/(2*limx*2*limy)
print(f"La densidad de electrones en el sistema es {densidad_e} electrones por picometro cuadrado")
print(f"El campo electrico fue de {E[0]} V/pm en dirección x")
print(f"El tiempo de relajación promedio tau fue {np.mean(prom_tau)} segundos")
print(f"El tiempo de relajación promedio a partir de la velocidad de deriva y el campo electrico {-promvx*m_e/(E*constants.elementary_charge)} segundos")
print(f"La conductividad del material a partir del tau fue de {N/(2*limx*2*limy)*constants.elementary_charge**2*np.mean(prom_tau)/constants.electron_mass}")
print(f"La conductividad del material a partir de la velocidad de deriva fue fue de {N/(2*limx*2*limy)*constants.elementary_charge**2*np.mean(prom_tau)/constants.electron_mass}")
print(f"La conductividad del material a partir J/E 1 {densidad_e*constants.elementary_charge*promvx/E}")
print(f"La conductividad del material a partir J/E 2 {contador/(2*limx*E)}")
print(f"El promedio de velocidades fue {velprom} m/s")
print(f"En promedio la velocidad de deriva fue {promvx} m/s")
print(f"La velocidad cuadratica media fue de {promv2} m/s y deberia ser {random_vel2(2)} m/s, la discrepancia porcentual es de {np.abs(promv2-random_vel2(2,promvx))/random_vel2(2,promvx)*100}%")

# Calcular las coordenadas x e y de la partícula en función del tiempo
#Lista de posiciones
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
#ani.save('animacion_DRUDE_V4.mp4', writer='ffmpeg')

plt.show()