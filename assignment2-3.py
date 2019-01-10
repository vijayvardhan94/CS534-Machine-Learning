import numpy as np
from math import log
import csv
import math
import copy

#Set p value (1,2,3,7,15)
p = 1
print(p)

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

def x_data_test(filename):
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
		add_x.append(tmp)
	x_to_array = np.array(add_x)			# x list convert to x array

	return x_to_array




#############Kernel Perceptron for train##############
def Kernel_Perceptron_Train(x, y):
	'''
		X = N*F, a = N*1, y = N*1 
		This function is for caculating alpha from train.csv
	'''
	N = x.shape[0]					#4888
	a = np.zeros(x.shape[0])		
	a_for_array = []

	
	for it in range(0,15):
		count = 0
		for i  in range(0, N): 
			u = 0
			for j in range(0, N):
				kp = (1 + np.dot(x[j], x[i].T)) ** p
				u = u + kp*a[j]*y[j] 
			if (y[i]*u) <= 0:
				count += 1
				a[i] += 1
		a_for_array.append(copy.deepcopy(a))
		accuracy = 1- count/N
		print("train:", accuracy)
	return a_for_array

	



#############Kernel Perceptron for validation##############
def Kernel_Perceptron_Valid(x, y, x_v, y_v):
	'''
		
		This function is for caculating accuracy and find alpha from valid.csv
	'''
	a = np.array(Kernel_Perceptron_Train(x_array_train, y_array_train)) 		# Train data's a
	N = x_v.shape[0]			#i
	N1 = x.shape[0]				#j


	for it in range(0,15):
		count = 0
		c = a[it]
		for i  in range(0, N): 
			u = 0
			for j in range(0, N1):
				kp = (1 + np.dot(x[j], x_v[i].T)) ** p
				u = u + kp* c[j]*y[j] 
			if y_v[i]*u <= 0:
				count += 1
		accuracy = 1 - (count/N)
		print("validation:", accuracy)
	return u




#####################Test Perceptron###################
# def Test_Perceptron(x, y, x_t):
# 	'''
# 		The best result is at p = 3 , iter = 4
# 		This function is for test y value in test.csv
# 	'''
# 	a = np.array(Kernel_Perceptron_Train(x_array_train, y_array_train)) 		# Train data's a
# 	N = x_t.shape[0]			#i 
# 	N1 = x.shape[0]				#j
# 	c = a[3]					#iter = 4
# 	u_list = list()

	

# 	for i in range(0, N): 
# 		u = 0
# 		for j in range(0, N1):
# 			kp = (1 + np.dot(x[j], x_t[i].T)) ** 3
# 			u = u + kp* c[j]*y[j]
# 		print("u:", np.sign(u))
# 		u_list.append(np.sign(u))

# 	return u_list





############Main Function############
#train.csv
y_array_train = y_data('pa2_train.csv')
x_array_train = x_data('pa2_train.csv')

#valid.csv
y_array_valid = y_data('pa2_valid.csv')
x_array_valid = x_data('pa2_valid.csv')

#test.csv
x_array_test = x_data_test('pa2_test_no_label.csv')

# Kernel_Perceptron_Train(x_array_train, y_array_train)

# Accuracy
Kernel_Perceptron_Valid(x_array_train, y_array_train, x_array_valid, y_array_valid)

# Output test value
# pred_y = Test_Perceptron(x_array_train, y_array_train, x_array_test)

rows = len(pred_y)
with open('kplabel.csv', 'a', newline='') as csvfile:
        
        writer = csv.writer(csvfile)
        for i in range(0, rows):
            writer.writerow([pred_y[i]])

