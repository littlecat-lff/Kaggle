# coding: utf-8
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def pic1():
	fig = plt.figure()
	ax = Axes3D(fig)
	
	x = np.arange(-4,4,1)
	y = np.arange(-4,4,1)
	X,Y = np.meshgrid(x,y)
	
#	R = np.sqrt(X**2 + Y**2)
#	Z = np.sin(R)
	
	R = X**2 + Y**2
	Z = R

	print X
	print Y
	print Z
	
	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
	#ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
	ax.contour(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
	
	plt.show()

def pic2():

	fig = plt.figure()
	ax = Axes3D(fig)
	
	x = np.arange(0, 200)
	y = np.arange(0, 100)
	x, y = np.meshgrid(x, y)
	z = np.random.randint(0, 200, size=(100, 200))%3
	print(z.shape)
	
	# ax.scatter(x, y, z, c='r', marker='.', s=50, label='')
	ax.plot_surface(x, y, z,label='')
	plt.show()

pic1()
