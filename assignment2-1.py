import numpy as np
from math import log
import matplotlib.pyplot as plt
import csv
import math


##########Read File###########
def y_data(filename):
	'''
		This function is for adding y value in list
	'''
	file = open(filename , 'r')
	add_y = []
	y = []
	a = csv.reader(file)
	for row in a : 
		add_y.append(row[0])
	for iy in add_y:
		integer = int(iy)
		if integer == 3:
			y.append(float(1))
		else:
			y.append(float(-1))
	y_to_array = np.array(y)
	return y_to_array

def x_data(filename):
	'''
		This function is for x value without y value
	'''
	file = open(filename, 'r') 
	add_x = []
	for row in file:
		tmp = []						#each row creates a list
		for x in row.split(','):		
			tmp.append(float(x.strip()))
		tmp.append(float(1))		
		add_x.append(tmp[1:])
	x_to_array = np.array(add_x)			# x list convert to x array

	return x_to_array

def test_x_data(filename):
	'''
		This function is for x value 
	'''
	file = open(filename, 'r') 
	add_x = []
	for row in file:
		tmp = []						#each row creates a list
		for x in row.split(','):		
			tmp.append(float(x.strip()))
		tmp.append(float(1))		
		add_x.append(tmp[:])
	x_to_array = np.array(add_x)			# x list convert to x array
	# print("x_to_array",x_to_array.shape)
	return x_to_array



#############Perceptron_Train##############
def Perceptron_Train(x, y):
	'''
		This function is for caculating accuracy for train.csv
	'''
	w = np.zeros(len(x[0]))		#number of values in Train.csv 
	it = 1					#iterate
	N = x.shape[0] 	
	accuracy = 0
	w_for_validation = []

	while it < 16:
		count = 0
		u = 0 					#initial x dot w.T
		for i in range(0, N):

			u = np.sign(np.dot(x[i], w.T))

			if (y[i]*u) <= 0:
				count += 1
				w = w + y[i]*x[i]
		w_for_validation.append(w)
		accuracy = 1-(count/N)
		# print(w_for_validation)
		print("train:", accuracy)		
		it = it + 1

	return w_for_validation


###########Perceptron_Validation################
def Perceptron_Validation(x, y, x_v, y_v, iter):
	'''
		This function is for caculating accuracy for valid.csv
	'''
	a = np.array(Perceptron_Train(x, y)) #w
	N = x_v.shape[0]
	accuracy = 0

	for it in range(0, iter):
		count = 0
		u = 0
		for i in range(0, N):
			u = np.sign(np.dot(x_v[i], a[it].T))
			if(y_v[i]*u) <= 0:
				count +=1
		accuracy = 1-(count/N)
		print("validation:", accuracy)
		it = it+1
	return a

###########Test value################
def test_value(w, x):

	y = np.zeros(x.shape[0])	
	for i in range(0, x.shape[0]):
		y[i] = np.sign(np.dot(x[i], w.T))
	return y

############Main Function############
#train.csv
y_array_train = y_data('pa2_train.csv')
x_array_train = x_data('pa2_train.csv')
#valid.csv
y_array_valid = y_data('pa2_valid.csv')
x_array_valid = x_data('pa2_valid.csv')
#test.csv
test_x_array = test_x_data('pa2_test_no_label.csv')

#iter = 15
w_array = Perceptron_Validation(x_array_train, y_array_train, x_array_valid, y_array_valid, 15)

#iter = 14
W_valid = w_array[13]

pred_y = test_value(W_valid, test_x_array)

rows = pred_y.shape[0]
with open('oplabel.csv', 'a', newline='') as csvfile:
        
        writer = csv.writer(csvfile)
        for i in range(0, rows):
            writer.writerow([pred_y[i]])