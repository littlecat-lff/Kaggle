# coding: utf8
#===============================================================================
# File  : ex5.py
# Desc  : 
#  
# Date  : 2016.11.04
#===============================================================================

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

def multi(x, dim):
	#return [x, math.pow(x,2), math.pow(x,3), math.pow(x,4), math.pow(x,5)]
	result = []
	for i in range(1, dim+1):
		result.append(math.pow(x,i))
	return result

def get_new_x_y(w, b):
	global more_x
	y_op = np.zeros(len(more_x))
	for i in range(0, len(more_x)):
		cur_x = more_x[i]
		multi_result = multi( cur_x, len(w) )
		y = b
		for j in range(0, len(multi_result)):
			y = y + w[j] * multi_result[j]

		y_op[i] = y
	return y_op

def step_gradient(xpoints, ypoints, multi_x, b, weight, lr):
	N = float(len(xpoints))

	g_m = np.zeros(len(multi_x[0]))
	g_b = 0

	new_b = 0
	new_weight = np.zeros(len(multi_x[0]))

	all_los = 0

	for i in range(0, len(xpoints)):
		x = xpoints[i]
		y = ypoints[i]

		# cal the loss
		cur_y = b*1.0
		#print "---------i:%d multi len:%d, weight len:%d------"%(i, len(multi_x), len(weight))

		for n in range(0, len(multi_x[i])):
			cur_y = cur_y + weight[n] * multi_x[i][n]
		loss = (cur_y-y) ** 2

		all_los = all_los + loss

		# update the diff
		gradient = 2 * (cur_y - y)

		#print "process x:%lf y:%lf cur_y:%lf gradient:%lf all_los:%lf"%(x, y, cur_y, gradient, all_los)

		g_b = g_b + gradient
		for j in range(0, len(weight)):
			g_m[j] = g_m[j] + gradient * multi_x[i][j] 

	new_b = b - lr * ( g_b * 1.0 ) / N
	for i in range(0, len(weight)):
		new_weight[i] = weight[i] - lr * ( g_m[i] * 1.0 / N )

	#print "N:%d lr:%lf loss:%lf"%(N, lr, all_los)

	return [new_b, new_weight, all_los]

def init():
	global ax1, xpoints, ypoints

def update(i):
	global xpoints,ypoints,multi_x,b,w,lr,line_gd
	for j in range(0,100):
		b, w, loss = step_gradient(xpoints, ypoints, multi_x, b, w, lr)
	print "----iter times:%d loss:%lf----"%(i*100, loss)
	y_op = get_new_x_y(w, b)
	# update the best line
	line_gd.set_data(more_x, y_op)

	return line_gd

def test():
	x = 2
	multi_x = multi(x, 6)
	print multi_x

# ------------------------------------------------------
# init 
fig = plt.figure()
ax1 = fig.add_subplot(111)

# load the data
xpoints = np.genfromtxt('./ex5Data/ex5Linx.dat')
ypoints = np.genfromtxt('./ex5Data/ex5Liny.dat')
ax1.scatter(xpoints, ypoints, c='red', marker='s')

# init the sgd
dim=5
b = 0
w = np.zeros(dim)
lr = 0.1
iter = 20000

multi_x = []
for j in range(0, len(xpoints)):
	multi_x.append(multi(xpoints[j], dim))

more_x = np.linspace(-1,1.0,100)
y_op = get_new_x_y(w, b)
line_gd, = ax1.plot(more_x, y_op, lw=1)

anim1=animation.FuncAnimation(fig, update, init_func=init,  frames=iter, interval=600)#, blit=True)
plt.show()
