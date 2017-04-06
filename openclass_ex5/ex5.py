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
import math


def multi(x):
	return [x, math.pow(x,2), math.pow(x,3), math.pow(x,4), math.pow(x,5)]

def step_gradient(xpoints, ypoints, multi_x, b, weight, lr):
	N = float(len(xpoints))

	g_m = np.zeros(5)
	g_b = 0

	new_b = 0
	new_weight = np.zeros(5)

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

	new_b = b - lr * g_b
	for i in range(0, len(weight)):
		new_weight[i] = weight[i] - lr * g_m[i]

	#print "N:%d lr:%lf loss:%lf"%(N, lr, all_los)

	return [new_b, new_weight, all_los]

def gradient_runner(xpoints, ypoints, multi_x, lr, iter):
	b = 0
	w = np.zeros(5)
	
	for i in range(0, iter):
		new_b, new_w, loss = step_gradient(xpoints, ypoints, multi_x, b, w, lr)
		print "----iter times:%d loss:%lf----"%(i, loss)
		b = new_b
		w = new_w

	return [b, w]

def run():
	# load the data
	xpoints = np.genfromtxt('./ex5Data/ex5Linx.dat')
	ypoints = np.genfromtxt('./ex5Data/ex5Liny.dat')

	plt.scatter(xpoints, ypoints, c='red', marker='s')

	multi_x = []
	for j in range(0, len(xpoints)):
		multi_x.append(multi(xpoints[j]))
	[b,w] = gradient_runner(xpoints, ypoints, multi_x, 0.1, 20000)

	print b
	print w

	more_x = np.linspace(-1,1.0,100)
	y_op = np.zeros(len(more_x))
	for i in range(0, len(more_x)):
		cur_x = more_x[i]
		multi_result = multi(cur_x)
		y = b
		for j in range(0, len(multi_result)):
			y = y + w[j] * multi_result[j]

		y_op[i] = y

	plt.plot(more_x, y_op, c='blue')

	plt.show()


def test():
	x = 2
	multi_x = multi(x)
	print multi_x

if __name__ == "__main__":
	run()
	#test()
