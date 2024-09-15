from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import scipy as sci
#!apt install ffmpeg

dt=0.01 #si es 0.01 la velocidad de las pelotas serán mucho más lentas
Nsteps=200
N=2 #numero de pelotas
kHertz=1.0e4
#Constantes del algoritmo de integración
xi =0.1786178958448091
Lambda = -0.2123418310626054
Chi = -0.06626458266981849
Um2labdau2 = (1-2*Lambda)/2
Um2chiplusxi = 1-2*(Chi+xi)
# initializing a figure in
# which the graph will be plotted
fig = plt.figure()
# marking the x-axis and y-axis
axis = plt.axes(xlim =(-10, 10), ylim =(-5, 5))

class Cuerpo:
  def __init__(self, x0, y0, Vx0, Vy0, m0, R0):
    self.m=m0
    self.R=R0
    self.r=np.array([x0,y0])
    self.V=np.array([Vx0,Vy0])
    self.F=np.zeros(2)

  def BorreFuerza(self):
    self.F=np.zeros(2)

  def SumeFuerza(self,dF):
    self.F=self.F+dF


  def Mueva_r(self, dt, coeficiente):
    self.r=self.r+(coeficiente*dt)*self.V

  def Mueva_V(self, dt, coeficiente):
    self.V=self.V+(coeficiente*dt)*self.F/self.m

class Colisionador:

  def __init__(self,N):
    self.N=N

  def CalcularFuerzasEntrePelotas(self,Pelota1,Pelota2):
    self.r21=np.array([0,0])
    self.r21=Pelota2.r-Pelota1.r
    self.d=np.linalg.norm(self.r21)
    self.s=(Pelota1.R+Pelota2.R)-self.d
    if self.s>0:
      self.n=np.array([0,0])
      self.n=self.r21*(1.0/self.d)

      self.F2=np.array([0,0])
      self.F2=self.n*(kHertz*(self.s**1.5))

      Pelota2.SumeFuerza(self.F2)
      Pelota1.SumeFuerza(-self.F2)

  def CalcularTodasLasFuerzas(self,Pelota):
    i=0
    for i in range(N):
      Pelota[i].BorreFuerza()
    i=0
    for i in range(N):
      for j in range(i+1):
        self.CalcularFuerzasEntrePelotas(Pelota1, Pelota2)



Pelota1=Cuerpo(0,0,0.5,0,1.0,0.5)
Pelota2=Cuerpo(3,0,-0.5,0,1.0,0.5)

Pelotas=[Pelota1, Pelota2]
choque=Colisionador(N)


line, = axis.plot([], [],color= 'purple',marker='o', linestyle='dashed',linewidth=2, markersize=12)
line2, = axis.plot([], [], color='lightgreen', marker='o', linestyle='dashed',linewidth=2, markersize=12)
def init():
  line.set_data([], [])
  return line,

xdata=np.zeros(Nsteps)
ydata=np.zeros(Nsteps)
vxdata=np.zeros(Nsteps)
vydata=np.zeros(Nsteps)

x2data=np.zeros(Nsteps)
y2data=np.zeros(Nsteps)
vx2data=np.zeros(Nsteps)
vy2data=np.zeros(Nsteps)

t=0

for t in range(0,Nsteps):
#x=x+xi*dx*v
  for i in range(0,N):
    Pelotas[i].Mueva_r(dt, xi)
#v=v+(1-2*Lambda)*dx*F/2
  choque.CalcularTodasLasFuerzas(Pelotas)
  for i in range(0,N):
    Pelotas[i].Mueva_V(dt, Um2labdau2)
#x=x+Chi*dx*v
  for i in range(0,N):
    Pelotas[i].Mueva_r(dt, Chi)
#v=v+Lambda*dx*F

  choque.CalcularTodasLasFuerzas(Pelotas)
  for i in range(0,N):
    Pelotas[i].Mueva_V(dt, Lambda)
#x=x+(1-2(Chi+xi))*dt*v
  for i in range(0,N):
    Pelotas[i].Mueva_r(dt, Um2chiplusxi)
#v=v+Lambda*dx*F
  choque.CalcularTodasLasFuerzas(Pelotas)
  for i in range(0,N):
    Pelotas[i].Mueva_V(dt, Lambda)
#x=x+Chi*dx*v
  for i in range(0,N):
    Pelotas[i].Mueva_r(dt, Chi)
#v=v+(1-2*Lambda)*dx*F/2
  choque.CalcularTodasLasFuerzas(Pelotas)
  for i in range(0,N):
    Pelotas[i].Mueva_V(dt, Um2labdau2)
#x=x+xi*dx*v
  for i in range(0,N):
    Pelotas[i].Mueva_r(dt, xi)

#Para guardar los datos y su animacion
  xdata[t]=Pelotas[0].r[0]
  ydata[t]=Pelotas[0].r[1]
  vxdata[t]=Pelotas[0].V[0]
  vydata[t]=Pelotas[0].V[1]
  x2data[t]=Pelotas[1].r[0]
  y2data[t]=Pelotas[1].r[1]
  vx2data[t]=Pelotas[1].V[0]
  vy2data[t]=Pelotas[1].V[1]
  if xdata[t]<=-10 or xdata[t]>=10:
    Pelotas[0].V[0]=-Pelotas[0].V[0]
    Pelotas[1].V[0]=-Pelotas[1].V[0]
  if ydata[t]<=-5 or ydata[t]>=5:
    Pelotas[0].V[1]=-Pelotas[0].V[1]
    Pelotas[1].V[1]=-Pelotas[1].V[1]


#Pasar los datos para animarlos y que se choquen con las paredes

"""
j=0
for j in range(0,Nsteps):
  xdata[j]=Pelotas[0].r[0]
  ydata[j]=Pelotas[0].r[1]
  vxdata[j]=Pelotas[0].V[0]
  vydata[j]=Pelotas[0].V[1]
  x2data[j]=Pelotas[1].r[0]
  y2data[j]=Pelotas[1].r[1]
  vx2data[j]=Pelotas[1].V[0]
  vy2data[j]=Pelotas[1].V[1]
  if xdata[j]<=-10 or xdata[j]>=10:
    Pelotas[0].V[0]=-Pelotas[0].V[0]
    Pelotas[1].V[0]=-Pelotas[1].V[0]
  if ydata[j]<=-5 or ydata[j]>=5:
    Pelotas[0].V[1]=-Pelotas[0].V[1]
    Pelotas[1].V[0]=-Pelotas[1].V[0]


"""
i=0
def animate(i):
  x=xdata[i]
  y=ydata[i]
  x2=x2data[i]
  y2=y2data[i]
  line.set_data(x, y)
  line2.set_data(x2,y2)
  return (line, line2,)

plt.plot(xdata, ydata,'b',x2data, y2data, 'r')
plt.show

anim=animation.FuncAnimation(fig, animate, init_func=init, frames=Nsteps, interval=20, blit=True)
anim.save('choque de dos pelotas (PEFRL) .mp4', fps=30)
