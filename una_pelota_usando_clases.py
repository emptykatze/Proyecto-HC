from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import scipy as sci
#!apt install ffmpeg

dt=1 #si es 0.01 la velocidad de las pelotas serán mucho más rápidas
Nsteps=2000
N=16 #numero de pelotas
# initializing a figure in
# which the graph will be plotted
fig = plt.figure()
# marking the x-axis and y-axis
axis = plt.axes(xlim =(-10, 10), ylim =(-5, 5))

class Cuerpo:
  def __init__(self, x0, y0, Vx0, Vy0, m0):
    self.m=m0
    self.r=np.array([x0,y0])
    self.V=np.array([Vx0,Vy0])

  def Muevase(self,dt):
    self.r=self.r+dt*self.V



xdata=np.zeros(Nsteps)
ydata=np.zeros(Nsteps)
vxdata=np.zeros(Nsteps)
vydata=np.zeros(Nsteps)

Pelota=Cuerpo(0,0,0.5,0.5,1)

line, = axis.plot([], [],color= 'purple',marker='o', linestyle='dashed',linewidth=2, markersize=12)

def init():
  line.set_data([], [])
  return line,

xdata[0]=Pelota.r[0]
ydata[0]=Pelota.r[1]

j=1

for j in range(0,Nsteps):
  Pelota.Muevase(dt)
  xdata[j]=Pelota.r[0]
  ydata[j]=Pelota.r[1]
  vxdata[j]=Pelota.V[0]
  vydata[j]=Pelota.V[1]
  if xdata[j]<=-10 or xdata[j]>=10:
    Pelota.V[0]=-Pelota.V[0]
  if ydata[j]<=-5 or ydata[j]>=5:
    Pelota.V[1]=-Pelota.V[1]



def animate(i):
  x=xdata[i]
  y=ydata[i]
  line.set_data(x, y)
  return line,



def animate(i):
  x=xdata[i]
  y=ydata[i]
  line.set_data(x, y)
  return line,

anim=animation.FuncAnimation(fig, animate, init_func=init, frames=Nsteps, interval=20, blit=False)
anim.save('Pelota_usando_clases.mp4', fps=30)
