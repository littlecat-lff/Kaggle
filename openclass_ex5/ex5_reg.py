# coding: utf8
#===============================================================================
# File  : ex5.py
# Desc  : 
#  
# Date  : 2016.11.04
# Author: jiuqian(81691)
# Email : jianhui.jjh@alibaba-inc.com
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

def step_gradient(xpoints, ypoints, multi_x, b, weight, lr, lambda_ratio=0):
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
		gradient = ( 2 * (cur_y - y) ) 

		#print "process x:%lf y:%lf cur_y:%lf gradient:%lf all_los:%lf"%(x, y, cur_y, gradient, all_los)

		g_b = g_b + gradient
		for j in range(0, len(weight)):
			g_m[j] = g_m[j] + gradient * multi_x[i][j] + lambda_ratio * weight[j] 

	new_b = b - lr * ( g_b * 1.0 / N )
	for i in range(0, len(weight)):
		new_weight[i] = weight[i] - lr * ( g_m[i] * 1.0 / N )

	#print "N:%d lr:%lf loss:%lf"%(N, lr, all_los)

	return [new_b, new_weight, all_los]

def init():
	global ax1, xpoints, ypoints

def update(i):
	global xpoints, ypoints, multi_x, b, w, lr, line_gd, line_gd_1, line_gd_10, regular_lambda
	for j in range(0,100):
		b_0,w_0,loss_0 = step_gradient(xpoints, ypoints, multi_x, b[0], w[0], lr[0], regular_lambda[0])
		b_1,w_1,loss_1 = step_gradient(xpoints, ypoints, multi_x, b[1], w[1], lr[1], regular_lambda[1])
		b_10,w_10,loss_10 = step_gradient(xpoints, ypoints, multi_x, b[2], w[2], lr[2], regular_lambda[2])
		b[0],w[0] = b_0, w_0
		b[1],w[1] = b_1, w_1
		b[2],w[2] = b_10, w_10
	print "----iter times:%d loss:%lf loss_1:%lf loss_10:%lf----"%(i*100, loss_0, loss_1, loss_10)
	y_op = get_new_x_y(w[0], b[0])
	y_op_1 = get_new_x_y(w[1], b[1])
	y_op_10 = get_new_x_y(w[2], b[2])
	# update the best line
	line_gd.set_data(more_x, y_op)
	line_gd_1.set_data(more_x, y_op_1)
	line_gd_10.set_data(more_x, y_op_10)

	return line_gd,line_gd_1,line_gd_10

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

# init the sgd parameter
dim=5
b = [0,0,0]
w = []
w.append(np.zeros(dim))
w.append(np.zeros(dim))
w.append(np.zeros(dim))

lr = [0.1, 0.001, 0.001]
iter = 20000
regular_lambda = [0, 1, 10] # the regularization ratio

for i in range(0,3):
	print b[i]

for i in range(0,3):
	print w[i]

for i in range(0,3):
	print regular_lambda[i]


#
multi_x = []
for j in range(0, len(xpoints)):
	multi_x.append(multi(xpoints[j], dim))

more_x = np.linspace(-1,1.0,100)
y_op = get_new_x_y(w[0], b[0])

line_gd, = ax1.plot(more_x, y_op, lw=1, label="lambda_0")
line_gd_1, = ax1.plot(more_x, y_op, lw=1, label="lambda_1")
line_gd_10, = ax1.plot(more_x, y_op, lw=1, label="lambda_10")

plt.legend()
plt.grid()
anim1=animation.FuncAnimation(fig, update, init_func=init,  frames=iter, interval=600)#, blit=True)
plt.show()
