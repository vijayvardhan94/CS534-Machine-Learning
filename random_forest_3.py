import time
import random as rd
import os
import numpy as np
from math import log
import csv
import math
import operator
from collections import OrderedDict

####################os.chdir('C:\\Users\\tianz\\Desktop\\machine learning\\\implementation 3')

class Node:
    def __init__(self, val=None, treedepth=None,  leftchildnode=None, rightchildnode=None, nodefeature=0, label=None):
        self.leftchildnode = leftchildnode
        self.rightchildnode = rightchildnode
        self.treedepth = treedepth
        self.val = val
        self.nodefeature = nodefeature
        self.label = label

class DecisionTree:
    def __init__(self):
        self.root = Node()
 
def read_data(filename):
	file = open(filename , 'r')
	add_o = []
	total = []
	for row in file:
		temp = []						
		for x in row.split(','):		
			temp.append(float(x.strip()))		
		add_o.append(temp)
	a = np.array(add_o)
	for x_v in a:
		str_xv = str(x_v[0])
		if str_xv == '5.0':
			p_str = str_xv.replace('5.0', '-1')
			x_v[0] = float(p_str)
		else:
			n_str = str_xv.replace('3.0', '1')
			x_v[0] = float(n_str)
		total.append(x_v)
		total_to_array = np.array(total)
	return total_to_array

def yvalues(filename):
	
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

def xvalues(filename):
	
	file = open(filename, 'r') 
	add_x = []
	for row in file:
		temp = []						
		for x in row.split(','):		
			temp.append(float(x.strip()))		
		add_x.append(temp[1:])
	x_to_array = np.array(add_x)			
	
	return x_to_array

def x_feature(x):
	
	x_T = x.T
	return x_T

def split_data(data_left, data_right):
 left_y = np.transpose(data_left)
 right_y = np.transpose(data_right)



 count_right_pos = (right_y[0]==1).sum()
 count_right_neg = (right_y[0]==-1).sum()
 count_left_pos = (left_y[0]==1).sum()
 count_left_neg = (left_y[0]==-1).sum()
 return count_left_neg,count_left_pos,count_right_neg, count_right_pos


def single_data_count(data):
	yvalues = np.transpose(data)
	count_y_pos = (yvalues[0]==1).sum()
	count_y_neg = (yvalues[0]==-1).sum()
	return count_y_neg,count_y_pos

def U_value(neg,pos):

	u = 1 - pow((pos/(pos+neg)), 2) - pow((neg/(pos+neg)), 2)
	return u 


def B_value(left_neg,left_pos, right_pos, right_neg, U_root):
	pb_l= (left_pos+left_neg)/(right_pos+right_neg+left_pos+left_neg)
	pb_r= (right_pos+right_neg)/(right_pos+right_neg+left_pos+left_neg)
	B_value = U_root - pb_l*U_value(left_neg,left_pos) - pb_r*U_value(right_neg,right_pos)
	return B_value

def best_B(x_array, pos, neg, size_x,select_feature):
	val, left_neg, left_pos, right_neg, right_pos, left_array, right_array = 0, 0, 0, 0, 0, None, None
	U_root = U_value(neg, pos)
	
	best_b = 0
	pre_y_value=0
	curr_y_value=0
	best_feature = 0
	temp_feature = 0
	temp_left_neg=0
	temp_left_pos=0
	temp_right_neg=0
	temp_right_pos=0
	count = 0
	
	for y in range(1,size_x):
		
		x_array_sorted = x_array[x_array[:,y].argsort()]
		
		pre_y_value=0
		curr_y_value=0
		count =0
		for i in range(1,np.size(x_array,0)):
			
			pre_y_value = curr_y_value
			
			curr_y_value = x_array_sorted[i][0]
			
			if pre_y_value != curr_y_value:
				count +=1
				temp_theda = x_array_sorted [i][y]
				temp_feature = select_feature[y]
				temp_left_neg, temp_left_pos, temp_right_neg, temp_right_pos = split_data(x_array_sorted[0:i],x_array_sorted[i:])
				
				if temp_left_neg==temp_left_pos==0 or temp_right_pos==temp_right_neg==0:
					temp_b=0
				else:
					temp_b = B_value(temp_left_neg, temp_left_pos, temp_right_pos, temp_right_neg, U_root )
				if temp_b > best_b:
					best_feature = temp_feature
					val = temp_theda
					left_pos = temp_left_pos
					left_neg = temp_left_neg
					right_neg = temp_right_neg
					right_pos = temp_right_pos
					left_array = x_array_sorted[0:i]
					right_array = x_array_sorted[i:]
					best_b = temp_b
					
	return  val, best_feature, left_neg, left_pos, right_neg, right_pos, left_array, right_array

def setlabel(pos, neg):
	if pos >= neg:
		return 1
	else:
		return -1


def create_node(root, treedepth, total_train,root_pos,root_neg, accur,size_x,select_feature):

	val, best_feature, left_neg, left_pos, right_neg, right_pos, left_array, right_array = best_B(total_train,root_pos,root_neg,size_x,select_feature)
	
	root.nodefeature = best_feature
	root.val = val
	root.treedepth = treedepth
	root.label = setlabel(root_pos, root_neg)
	
	accur[root.treedepth+1] = accur.setdefault(root.treedepth+1, 0) + min(left_neg, left_pos) + min(right_neg, right_pos)

	if root.treedepth <= max_depth:
		if left_neg != 0 and left_pos != 0:
			root. leftchildnode = Node()
			root. leftchildnode = create_node(root.leftchildnode, treedepth+1, left_array,left_pos,left_neg, accur, size_x,select_feature)
		elif left_neg != 0 or left_pos != 0:
			root.leftchildnode = Node()
			root.leftchildnode.treedepth = treedepth+1
			root.leftchildnode.label = setlabel(left_pos, left_neg)

		if right_neg != 0 and right_pos != 0:
			root.rightchildnode = Node()
			root.rightchildnode = create_node(root.rightchildnode, treedepth+1, right_array,right_pos,right_neg, accur, size_x,select_feature)
		elif right_neg != 0 or right_pos != 0:
			root.leftchildnode = Node()
			root.leftchildnode.treedepth = treedepth+1
			root.leftchildnode.label = setlabel(right_pos, right_neg)

	
	return root

def compute_accur(accur, leng):
	for key in accur:
		print('depth: ', key, " ;  accuracy:", 1-(accur[key]/leng))

def v_setLabel(pos, neg, label):
	#error rate
	if label == 1:
		return neg
	else:
		return pos

def sort_cut(total_valid, nodefeature, val):

	x_array_sorted = total_valid[total_valid[:,nodefeature].argsort()]
	x_array_sorted_t = np.transpose(x_array_sorted)

	split_point = (x_array_sorted_t[nodefeature] < val).sum()

	left_array = x_array_sorted[0:split_point]
	right_array = x_array_sorted[split_point:]

	count_left_neg,count_left_pos,count_right_neg, count_right_pos = split_data(left_array, right_array)
	
	return left_array, right_array, count_left_neg,count_left_pos,count_right_neg, count_right_pos


def validation(root, depth, total_valid, v_accur, root_pos, root_neg):
	if treedepth <= max_depth:
		if root.val ==  None:
			# leaf
			for j in range(treedepth+1, max_depth+1):
				v_accur[j] = v_accur.setdefault(j, 0) + v_setLabel(root_pos, root_neg, root.label)
		else:

			count_left_neg,count_left_pos,count_right_neg, count_right_pos, left_array, right_array = 0,0,0,0,[],[]
			
			left_array, right_array, count_left_neg,count_left_pos,count_right_neg, count_right_pos = sort_cut(total_valid, root.feature, root.val)

			if root.leftchildnode != None or root.rightchildnode!= None:
				v_accur[treedepth+1] = v_accur.setdefault(treedepth+1, 0) + v_setLabel(count_right_pos, count_right_neg, root.rchild.label) + v_setLabel(count_left_pos, count_left_neg, root.leftchildnode.label)
				validation(root.leftchildnode, treedepth+1, left_array, v_accur, count_left_pos, count_left_neg)
				validation(root.rightchildnode, treedepth+1, right_array, v_accur, count_right_pos, count_right_neg)
	
	return root

def output_pred_y(total_data, root):

	pred_y = np.zeros(total_data.shape[0])
	find_leaf(total_data, root, pred_y)

	return pred_y

def find_leaf(total_data, root, pred_y):

	if root.leftchildnode == None or root.rightchildnode == None:
		idx_array = get_index(total_data)
		for idx in idx_array:
			pred_y[int(idx)] = root.label

	else:
		left_array, right_array, _,_,_,_ = sort_cut(total_data, root.nodefeature, root.val)

		find_leaf(left_array, root.leftchildnode, pred_y)
		find_leaf(right_array, root.rightchildnode, pred_y)

def get_index(data):
	trans_data = np.transpose(data)
	idx = trans_data[-1]
	
	return idx

def accu_calculate(pred_y, y_array):
	calculate_y = np.add(pred_y,y_array)
	error_value=(calculate_y==0).sum()
	accu = (pred_y.shape[0]-error_value)/pred_y.shape[0]
	return accu


y_array_train = yvalues('pa3_train_reduced.csv')
x_array_train = xvalues('pa3_train_reduced.csv')
total_train = read_data('pa3_train_reduced.csv')

y_array_valid = yvalues('pa3_valid_reduced.csv')
x_array_valid = xvalues('pa3_valid_reduced.csv')
total_valid = read_data('pa3_valid_reduced.csv')

length = list(range(len(x_array_train)))

dic_data = list(zip(length, total_train))

data_idx=[]
for i in range(0, 4888):
	data_idx.append([i])
total_train = np.append(total_train, data_idx, axis = 1)

vdata_idx=[]
for i in range(0, 1629):
 vdata_idx.append([i])
total_valid = np.append(total_valid, vdata_idx, axis = 1)

length = list(range(len(x_array_train)))

dic_data = list(zip(length, total_train))

tree_number = [1,2,5,10,25]
num_feature = 10
tree_list = list()
print("##########below is the accuracies for m=",num_feature,"########")
for i in tree_number:
	
	tree_list = list()
	pred_y_train = np.zeros(total_train.shape[0])
	pred_y_valid = np.zeros(total_valid.shape[0])
	for j in range(i):
		
		range_feature = list(range(1,101))
		range_example = list(range(4888))
		select_feature = np.random.choice(range_feature,num_feature,replace=False)
		select_feature = np.insert(select_feature,0,0)
		select_feature = np.insert(select_feature,num_feature+1,101)
		select_example = np.random.choice(range_example,4888,replace=True)

		
		modified_x_array = total_train[:,select_feature]
		input_x_array = modified_x_array[select_example]
		
		input_x_array_t = np.transpose(input_x_array)

	
		root_pos = (input_x_array_t[0]==1).sum()
		root_neg = (input_x_array_t[0]==-1).sum()
		tree_list.append(DecisionTree())
		accur = dict()
		tree_list[j].root.treedepth = 0
		max_depth = 9
		size_x = input_x_array.shape[1]
		
		create_node(tree_list[j].root, 0, input_x_array,root_pos,root_neg, accur, size_x-1,select_feature)
		
		pred_y_train_temp = output_pred_y(total_train, tree_list[j].root)
		pred_y_train = np.add(pred_y_train,pred_y_train_temp)
		
		pred_y_valid_temp = output_pred_y(total_valid, tree_list[j].root)
		pred_y_valid = np.add(pred_y_valid,pred_y_valid_temp)
		
	pred_y_train = np.sign(pred_y_train)
	pred_y_valid = np.sign(pred_y_valid)
	
	for k in range(pred_y_train.shape[0]):
		
		if pred_y_train[k] == 0:
			pred_y_train[k] = rd.choice([1,-1])
			
	for k in range(pred_y_valid.shape[0]):
		
		if pred_y_valid[k] == 0:
			pred_y_valid[k] = rd.choice([1,-1])
			
	accu_train = accu_calculate(pred_y_train,y_array_train)
	accu_valid = accu_calculate(pred_y_valid,y_array_valid)
	print("n=", i)
	print("train accuracy: ",accu_train)
	print("validation accuracy: ",accu_valid)
	tree_number = [1,2,5,10,25]
num_feature = 20
tree_list = list()
print("##########below is the accuracies for m=",num_feature,"########")
for i in tree_number:
	
	tree_list = list()
	pred_y_train = np.zeros(total_train.shape[0])
	pred_y_valid = np.zeros(total_valid.shape[0])
	for j in range(i):
		
		range_feature = list(range(1,101))
		range_example = list(range(4888))
		select_feature = np.random.choice(range_feature,num_feature,replace=False)
		select_feature = np.insert(select_feature,0,0)
		select_feature = np.insert(select_feature,num_feature+1,101)
		select_example = np.random.choice(range_example,4888,replace=True)

		modified_x_array = total_train[:,select_feature]
		input_x_array = modified_x_array[select_example]


		
		input_x_array_t = np.transpose(input_x_array)

	
		root_pos = (input_x_array_t[0]==1).sum()
		root_neg = (input_x_array_t[0]==-1).sum()
		tree_list.append(DecisionTree())
		accur = dict()
		tree_list[j].root.treedepth = 0
		max_depth = 9
		size_x = input_x_array.shape[1]
		
		create_node(tree_list[j].root, 0, input_x_array,root_pos,root_neg, accur, size_x-1,select_feature)
		
		pred_y_train_temp = output_pred_y(total_train, tree_list[j].root)
		pred_y_train = np.add(pred_y_train,pred_y_train_temp)
		
		pred_y_valid_temp = output_pred_y(total_valid, tree_list[j].root)
		pred_y_valid = np.add(pred_y_valid,pred_y_valid_temp)
		
	pred_y_train = np.sign(pred_y_train)
	pred_y_valid = np.sign(pred_y_valid)
	
	for k in range(pred_y_train.shape[0]):
		
		if pred_y_train[k] == 0:
			pred_y_train[k] = rd.choice([1,-1])
			
	for k in range(pred_y_valid.shape[0]):
		
		if pred_y_valid[k] == 0:
			pred_y_valid[k] = rd.choice([1,-1])
			
	accu_train = accu_calculate(pred_y_train,y_array_train)
	accu_valid = accu_calculate(pred_y_valid,y_array_valid)
	print("n=", i)
	print("train accuracy: ",accu_train)
	print("validation accuracy: ",accu_valid)




num_feature = 50
tree_list = list()
print("##########below is the accuracies for m=",num_feature,"########")
for i in tree_number:
	
	tree_list = list()
	pred_y_train = np.zeros(total_train.shape[0])
	pred_y_valid = np.zeros(total_valid.shape[0])
	for j in range(i):
		
		range_feature = list(range(1,101))
		range_example = list(range(4888))
		select_feature = np.random.choice(range_feature,num_feature,replace=False)
		select_feature = np.insert(select_feature,0,0)
		select_feature = np.insert(select_feature,num_feature+1,101)
		select_example = np.random.choice(range_example,4888,replace=True)

		modified_x_array = total_train[:,select_feature]
		input_x_array = modified_x_array[select_example]


		
		input_x_array_t = np.transpose(input_x_array)

		
		root_pos = (input_x_array_t[0]==1).sum()
		root_neg = (input_x_array_t[0]==-1).sum()
		tree_list.append(DecisionTree())
		accur = dict()
		tree_list[j].root.treedepth = 0
		max_depth = 9
		size_x = input_x_array.shape[1]
		
		create_node(tree_list[j].root, 0, input_x_array,root_pos,root_neg, accur, size_x-1,select_feature)
		
		pred_y_train_temp = output_pred_y(total_train, tree_list[j].root)
		pred_y_train = np.add(pred_y_train,pred_y_train_temp)
		
		pred_y_valid_temp = output_pred_y(total_valid, tree_list[j].root)
		pred_y_valid = np.add(pred_y_valid,pred_y_valid_temp)
		
	pred_y_train = np.sign(pred_y_train)
	pred_y_valid = np.sign(pred_y_valid)
	
	for k in range(pred_y_train.shape[0]):
		
		if pred_y_train[k] == 0:
			pred_y_train[k] = rd.choice([1,-1])
			
	for k in range(pred_y_valid.shape[0]):
		
		if pred_y_valid[k] == 0:
			pred_y_valid[k] = rd.choice([1,-1])
			
	accu_train = accu_calculate(pred_y_train,y_array_train)
	accu_valid = accu_calculate(pred_y_valid,y_array_valid)
	print("n=", i)
	print("train accuracy: ",accu_train)
	print("validation accuracy: ",accu_valid)

