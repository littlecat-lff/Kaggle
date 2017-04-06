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
from mpl_toolkits.mplot3d import Axes3D
import math

def step_gradient(w, lr):

	global pos_x,pos_y,neg_x,neg_y,dim,map_features_pos,map_features_neg
	# count the loss
	pos_num,neg_num = len(pos_x), len(neg_x)
	all_num = pos_num + neg_num
	all_los = 0

	fea_size = len(w)
	g_fea = np.zeros(fea_size)

	for i in range(0, pos_num):
#		cur_x,cur_y = pos_x[i],pos_y[i]
#		cur_f = map_feature([cur_x, cur_y], dim)
		cur_f = map_features_pos[i]
		cur_v = sum(w * cur_f)

		h_xy = 1.0/( 1 + math.pow(math.e, -cur_v))
		all_los = all_los - math.log(h_xy)

		for m in range(0,fea_size):
			cur_fea = cur_f[m]
			g_fea[m] = g_fea[m] + (h_xy - 1) * cur_fea

	for i in range(0, neg_num):
#		cur_x,cur_y = neg_x[i],neg_y[i]
#		cur_f = map_feature([cur_x, cur_y], dim)
		cur_f = map_features_neg[i]
		cur_v = sum(w * cur_f)

		h_xy = 1.0/( 1 + math.pow(math.e, -cur_v))
		all_los = all_los - math.log(1-h_xy)

		for m in range(0,fea_size):
			cur_fea = cur_f[m]
			g_fea[m] = g_fea[m] + h_xy * cur_fea
			#print "---------i:%d multi len:%d, weight len:%d------"%(i, len(multi_x), len(weight))

	new_weight = np.zeros(fea_size)
	for i in range(0, fea_size):
		new_weight[i] = w[i] - lr * 1.0 * g_fea[i] / all_num

	return [new_weight, all_los]

# --------------------------------------

def map_feature(point_in, dim):
	featues = [1]
	u = point_in[0]
	v = point_in[1]
	for i in range(1, dim+1):
		for j in range(0, i+1):
			t_fea = math.pow(u, i-j) * math.pow(v, j) 
			featues.append(t_fea)

	return featues

def function_x_y(X,Y, w, dim):
	xrow = len(X)
	xcol = len(X[0])
	yrow = len(Y)
	ycol = len(Y[0])

	print "xrow:%d xcol:%d yrow:%d ycol:%d dim:%d"%(xrow, xcol, yrow, ycol, dim)

	z = []

	# create z one row by one row
	for row in range(0, xrow):
		row_z = []
		for col in range(0, xcol):
			cur_x = X[row][col]
			cur_y = Y[row][col]
			cur_f = map_feature([cur_x, cur_y], dim)
			cur_z = sum(w * cur_f)

			row_z.append(cur_z)
		z.append(row_z)	

	return z

def init():
	global ax,X,Y
	return 1

def update(i):
	global X,Y,w,lr,ax,plt,line_loss,loss_array
	w,loss = step_gradient(w, lr)
	print "iter:%d and loss:%f"%(i,loss)

	loss_array.append(loss)

#	ssrecord = [i for i in range(len(loss_array))]
#	line_loss.set_data(ssrecord, loss_array)

	#Z = function_x_y(X, Y, w, dim)
	#print "zlen:%d xlen:%d ylen:%d"%(len(Z), len(X), len(Y))
	#ax.contour(X,Y,Z,0)

	return []

# ------------------------------------------------------

# load the data
xpoints = np.genfromtxt('./ex5Data/ex5Logx.dat', delimiter=',')
ylabels = np.genfromtxt('./ex5Data/ex5Logy.dat')

dim = 6
test = map_feature([2,3], dim)
print "len:%d"%(len(test))
print test

pos_x, pos_y, pos_z = [],[],[]
neg_x, neg_y, neg_z = [],[],[]
for i in range(0, len(ylabels)):
	if ylabels[i] == 1:
		pos_x.append(xpoints[i][0])
		pos_y.append(xpoints[i][1])
		pos_z.append(0)
	if ylabels[i] == 0:
		neg_x.append(xpoints[i][0])
		neg_y.append(xpoints[i][1])
		neg_z.append(0)

map_features_pos = []
map_features_neg = []

pos_num,neg_num = len(pos_x), len(neg_x)
for i in range(0, pos_num):
	cur_x,cur_y = pos_x[i],pos_y[i]
	cur_f = map_feature([cur_x, cur_y], dim)
	map_features_pos.append(cur_f)

for i in range(0, neg_num):
	cur_x,cur_y = neg_x[i],neg_y[i]
	cur_f = map_feature([cur_x, cur_y], dim)
	map_features_neg.append(cur_f)


### init the sgd
w = np.zeros( len(test) ) # this is the feature size
print w
lr = 0.4
iter = 100
loss_array = []

#wstr="4.69664767 4.03969212 5.39581616 -4.50093383 -6.99664598 -9.74631309 1.48321431 -1.18918189 -0.69625571 -2.4605478 -5.75889563 2.89679552 -5.03967458 -2.77867807 -4.44554718 -3.18548369 -0.66013359 3.21248437 -3.74398468 -3.32500624 2.06246023 -7.01535764 1.22982566 -1.54841736 2.16881191 -3.76018835 -3.11414547 1.12153693"
wstr="4.80156655 4.0540936 5.79462572 -4.32239013 -7.51385764 -9.86473876 1.98105608 -0.6407175 -0.52413835 -3.47850473 -5.68115918 3.58637658 -5.95282726 -2.74427908 -4.95624303 -3.93347941 -0.96318488 4.77251089 -4.724892 -4.02669348 2.72172172 -8.1839069 0.90862825 -1.45109265 3.13582214 -4.91195175 -3.69616293 1.66019146"
#wstr="4.53554566 3.66097436 6.21308451 -3.39698709 -7.61456363 -7.40180597 4.35910701 2.7540908 -0.77223114 -6.14391438 -3.51264578 3.81754321 -9.75255771 -1.34098856 -7.93947279 -6.88015449 -4.12194514 11.48544458 -8.32807772 -7.70027244 5.75380845 -13.6327867 -3.8663265 -0.38124196 7.41232582 -10.00724125 -6.46276027 1.42371736"

ps =  wstr.split(" ")
for i in range(0,len(ps)):
	w[i] = float(ps[i])
print w

more_x = np.linspace(-1.0,1.5,200)
X,Y = np.meshgrid(more_x, more_x)
Z = function_x_y(X, Y, w, dim)

for i in range(0, iter):
	if i > 50000:
		lr = 0.2
	w,loss = step_gradient(w, lr)
	loss_array.append(loss)
	if i%5000==0:
		print "iter:",i
		print w
		print "iter:%d and loss:%f"%(i,loss)

Z = function_x_y(X, Y, w, dim)


## ------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax.scatter(pos_x, pos_y, c='red', marker='+')
ax.scatter(neg_x, neg_y, c='yellow', marker='o')

ssrecord = [i for i in range(len(loss_array))]
ax2.set_ylim([0,100])
ax2.set_xlim([0, len(loss_array) + 10])

line_loss, = ax2.plot(ssrecord, loss_array, lw=1, c='red')

# draw the pic
#contour_sgd = ax.contour(X,Y,Z,0)
#anim1=animation.FuncAnimation(fig, update, init_func=init,  frames=iter, interval=60)#, blit=True)

ax.contour(X,Y,Z,[0])

plt.grid()
plt.show()






##------------------------------------------------------------
#ax = Axes3D(fig)
#ax.scatter(pos_x, pos_y, pos_z, c='red', marker='+')
#ax.scatter(neg_x, neg_y, neg_z, c='yellow', marker='o')
##ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
#ax.contour(X, Y, Z, zdir='z', offset=0, cmap=plt.cm.hot) 
#ax.set_zlim(0,1)
