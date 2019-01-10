import numpy as np
from math import log
# import matplotlib.pyplot as plt
import csv


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

#############Average Perceptron##############
def Avg_Perceptron(x_t, y_t, x_v, y_v):
    '''
        
    '''
    avg_w = np.zeros(x_t.shape[1])
    w = np.zeros(len(x_t[0]))
    s = 0
    c = 0
    u = 0
    itr = 1

    N_t = x_t.shape[0]
    N_v = x_v.shape[0]

    while itr < 16:
        print("===itr = ", itr,"====")
        c = 0
        acc_t = 0
        acc_v = 0
        accuracy_train = 0
        accuracy_valid = 0
        for i in range(0, N_t):
            u = np.sign(np.dot(x_t[i], w.T))
            if (y_t[i]*u) <= 0:
                if s+c > 0:
                    avg_w = (s*avg_w + c*w)/(s+c)
                s += c
                w = w + y_t[i]*x_t[i]
                c = 0
                acc_t += 1
            else:
                c += 1
        accuracy_train = acc_t/N_t
        print("=========train=======")
        print(1 - accuracy_train)


        for r in range(0, N_v):
            u = np.sign(np.dot(x_v[r], avg_w.T))
            if (y_v[r]*u) <= 0:
                acc_v += 1
        accuracy_valid = acc_v/N_v
        print("=========valid=======")
        print(1 - accuracy_valid)
        

        itr = itr + 1
        
    if c > 0:
        avg_w = (np.dot(s, avg_w) + np.dot(c, w))/(s+c)
                
    return avg_w


############Main Function############

#train.csv
y_array = y_data('pa2_train.csv')
x_array = x_data('pa2_train.csv')
#valid.csv
v_y_array = y_data('pa2_valid.csv')
v_x_array = x_data('pa2_valid.csv')
#test.csv
test_x_array = test_x_data('pa2_test_no_label.csv')


Avg_Perceptron(x_array, y_array, v_x_array, v_y_array)