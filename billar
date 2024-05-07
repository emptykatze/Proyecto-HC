from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

!apt install ffmpeg
# initializing a figure in
# which the graph will be plotted
fig = plt.figure()

# marking the x-axis and y-axis
axis = plt.axes(xlim =(-10, 10),
                ylim =(-5, 5))

# initializing a line variable
line, = axis.plot([], [], color='whitesmoke', marker='o', linestyle='dashed',linewidth=2, markersize=12)
line2, = axis.plot([], [], color='r', marker='o', linestyle='dashed',linewidth=2, markersize=12)
line3, = axis.plot([], [], color='b', marker='o', linestyle='dashed',linewidth=2, markersize=12)
line4, = axis.plot([], [], color='g', marker='o', linestyle='dashed',linewidth=2, markersize=12)
line5, = axis.plot([], [], color='y', marker='o', linestyle='dashed',linewidth=2, markersize=12)
line6, = axis.plot([], [], color='c', marker='o', linestyle='dashed',linewidth=2, markersize=12)
line7, = axis.plot([], [], color='m', marker='o', linestyle='dashed',linewidth=2, markersize=12)
line8, = axis.plot([], [], color='k', marker='o', linestyle='dashed',linewidth=2, markersize=12)
line9, = axis.plot([], [], color='orange', marker='o', linestyle='dashed',linewidth=2, markersize=12)
line10, = axis.plot([], [], color='purple', marker='o', linestyle='dashed',linewidth=2, markersize=12)
line11, = axis.plot([], [], color='brown', marker='o', linestyle='dashed',linewidth=2, markersize=12)
line12, = axis.plot([], [], color='lightgreen', marker='o', linestyle='dashed',linewidth=2, markersize=12)
line13, = axis.plot([], [], color='navy', marker='o', linestyle='dashed',linewidth=2, markersize=12)
line14, = axis.plot([], [], color='darkred', marker='o', linestyle='dashed',linewidth=2, markersize=12)
line15, = axis.plot([], [], color='coral', marker='o', linestyle='dashed',linewidth=2, markersize=12)
line16, = axis.plot([], [], color='beige', marker='o', linestyle='dashed',linewidth=2, markersize=12)
# data which the line will
# contain (x, y)
def init():
    line.set_data([], [])
    return line,
l1_x=np.linspace(-10, 10, 20)
l1_y=np.zeros((20))+5

l2_x=np.linspace(-10, 10, 20)
l2_y=np.zeros((20))-5

l3_x=np.zeros((20))-10
l3_y=np.linspace(-5, 5, 20)

l4_x=np.zeros((20))+10
l4_y=np.linspace(-5, 5, 20)


plt.plot(l1_x, l1_y, 'b')
plt.plot(l2_x, l2_y, 'b')
plt.plot(l3_x, l3_y, 'b')
plt.plot(l4_x, l4_y, 'b')

x=np.zeros(2000)
y=np.zeros(2000)
t=np.zeros(2000)
h=1

x[0]=-3
y[0]=0
vx=0.5
vy=0.5
j=1

for j in range(0,2000):
  x[j]=x[j-1]+h*vx
  y[j]=y[j-1]+h*vy
  if x[j]<=-10 or x[j]>=10:
    vx=-vx
  if y[j]<=-5 or y[j]>=5:
    vy=-vy

def creador_de_pelotas(px0, py0, pv_x0, pv_y0, h):
  pelota = np.array([],[])
  px=np.zeros(2000)
  py=np.zeros(2000)
  h=1
  px[0]=px0
  py[0]=py0
  pvx=pv_x0
  pvy=pv_y0

  j=1
  for j in range(0,2000):
    px[j]=px[j-1]+h*pvx
    py[j]=py[j-1]+h*pvy
    if px[j]<=-10 or px[j]>=10:
      pvx=-pvx
    if py[j]<=-5 or py[j]>=5:
      pvy=-pvy
  pelota_x= px
  pelota_y= py
  return (pelota_x, pelota_y)

pelota_2x, pelota_2y= creador_de_pelotas(0, 0, 0.1, 0, 1)
pelota_3x, pelota_3y= creador_de_pelotas(1.0, 0.5, -0.1, 0, 1)
pelota_4x, pelota_4y= creador_de_pelotas(1.0, -0.5, 0.1, 0, 1)
pelota_5x, pelota_5y= creador_de_pelotas(2.0, 0, 0.2, -0.1, 1)
pelota_6x, pelota_6y= creador_de_pelotas(2.0, 1.0, 0.31, -0.4, 1)
pelota_7x, pelota_7y= creador_de_pelotas(2.0, -1.0, 0.1, 0, 1)
pelota_8x, pelota_8y= creador_de_pelotas(3.0, 0.5, -3.1, 5.1, 1)
pelota_9x, pelota_9y= creador_de_pelotas(3.0, -0.5, 1.1, -1, 1)
pelota_10x, pelota_10y= creador_de_pelotas(3.0, 1.5, 0.5, -0.5, 1)
pelota_11x, pelota_11y= creador_de_pelotas(3.0, -1.5, -0.7, 0.8, 1)
pelota_12x, pelota_12y= creador_de_pelotas(4.0, 0, 0.3, 3.4, 1)
pelota_13x, pelota_13y= creador_de_pelotas(4.0, 1.0, 0.12, 4, 1)
pelota_14x, pelota_14y= creador_de_pelotas(4.0, -1.0, 0.6, 2, 1)
pelota_15x, pelota_15y= creador_de_pelotas(4.0, 2.0, 2, -2.5, 1)
pelota_16x, pelota_16y= creador_de_pelotas(4.0, -2.0, 0.4, 0.8, 1)

def animate(i):
  j=0
  x_a=x[i]
  y_a=y[i]

  x2_a= pelota_2x[i]
  y2_a= pelota_2y[i]

  x3_a= pelota_3x[i]
  y3_a= pelota_3y[i]

  x4_a= pelota_4x[i]
  y4_a= pelota_4y[i]

  x5_a= pelota_5x[i]
  y5_a= pelota_5y[i]

  x6_a= pelota_6x[i]
  y6_a= pelota_6y[i]

  x7_a= pelota_7x[i]
  y7_a= pelota_7y[i]

  x8_a= pelota_8x[i]
  y8_a= pelota_8y[i]

  x9_a= pelota_9x[i]
  y9_a= pelota_9y[i]

  x10_a= pelota_10x[i]
  y10_a= pelota_10y[i]

  x11_a= pelota_11x[i]
  y11_a= pelota_11y[i]

  x12_a= pelota_12x[i]
  y12_a= pelota_12y[i]

  x13_a= pelota_13x[i]
  y13_a= pelota_13y[i]

  x14_a= pelota_14x[i]
  y14_a= pelota_14y[i]

  x15_a= pelota_15x[i]
  y15_a= pelota_15y[i]

  x16_a= pelota_16x[i]
  y16_a= pelota_16y[i]

  line.set_data(x_a, y_a)
  line2.set_data(x2_a,y2_a)
  line3.set_data(x3_a,y3_a)
  line4.set_data(x4_a,y4_a)
  line5.set_data(x5_a,y5_a)
  line6.set_data(x6_a,y6_a)
  line7.set_data(x7_a,y7_a)
  line8.set_data(x8_a,y8_a)
  line9.set_data(x9_a,y9_a)
  line10.set_data(x10_a,y10_a)
  line11.set_data(x11_a,y11_a)
  line12.set_data(x12_a,y12_a)
  line13.set_data(x13_a,y13_a)
  line14.set_data(x14_a,y14_a)
  line15.set_data(x15_a,y15_a)
  line16.set_data(x16_a,y16_a)
  return (line, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12, line13, line14, line15, line16,)

anim = FuncAnimation(fig, animate, init_func = init,
                     frames = len(x), interval = 20, blit = False)

f = r"Billar_Prueba 16 pelotas.mp4"
writervideo = animation.FFMpegWriter(fps=60)
anim.save(f, writer=writervideo)
