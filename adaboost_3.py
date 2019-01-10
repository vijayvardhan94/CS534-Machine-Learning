import csv
import math
import operator
from collections import OrderedDict
import time
import random as rd
import numpy as np
from math import log


class Node:
    def __init__(self, val=None, treedepth=None, leftchildnode=None, rightchildnode=None	, nodefeature=0, label=None):
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

 left_pos_idx = np.where((data_left.T)[0]==1)[0]
 left_neg_idx = np.where((data_left.T)[0]==-1)[0]
 right_pos_idx = np.where((data_right.T)[0]==1)[0]
 right_neg_idx = np.where((data_right.T)[0]==-1)[0]

 count_right_pos = np.sum(D[right_pos_idx])
 count_right_neg = np.sum(D[right_neg_idx])
 count_left_pos = np.sum(D[left_pos_idx])
 count_left_neg = np.sum(D[left_neg_idx])
 return count_left_neg,count_left_pos,count_right_neg, count_right_pos


def U_value(neg,pos):
	u = 1 - pow((pos/(pos+neg)), 2) - pow((neg/(pos+neg)), 2)
	return u 

def B_value(left_neg,left_pos, right_pos, right_neg, U_root):
	pb_l= (left_pos+left_neg)/(right_pos+right_neg+left_pos+left_neg)
	pb_r= (right_pos+right_neg)/(right_pos+right_neg+left_pos+left_neg)
	B_value = U_root - pb_l*U_value(left_neg,left_pos) - pb_r*U_value(right_neg,right_pos)
	return B_value

def best_B(x_array, pos, neg, size_x):
	val, left_neg, left_pos, right_neg, right_pos, left_array, right_array = 0, 0, 0, 0, 0, None, None
	U_root = U_value(neg, pos)
	
	best_b = 0
	tmp_theda_index = [0,0]
	theda_index= [0,0]
	pre_y_value=0
	curr_y_value=0
	left_y =[]
	best_feature = 0
	temp_feature = 0
	temp_left_neg=0
	temp_left_pos=0
	temp_right_neg=0
	temp_right_pos=0
	count = 0
	for y in range(1,size_x):	
		x_array_sorted = x_array[x_array[:,y].argsort()]
		curr_y_value=0
		count =0
		for i in range(1,np.size(x_array,0)):
			
			pre_y_value = curr_y_value
		
			curr_y_value = x_array_sorted[i][0]
			
			if pre_y_value != curr_y_value:
				count +=1
				temp_theda = x_array_sorted [i][y]
				temp_feature = y
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


def create_node(root, treedepth, total_train,root_pos,root_neg, accur,size_x):

	val, best_feature, left_neg, left_pos, right_neg, right_pos, left_array, right_array = best_B(total_train,root_pos,root_neg,size_x)
	
	root.nodefeature = best_feature
	root.val = val
	root.treedepth = treedepth
	root.label = setlabel(root_pos, root_neg)
	
	accur[root.treedepth+1] = accur.setdefault(root.treedepth+1, 0) + min(left_neg, left_pos) + min(right_neg, right_pos)

	if root.treedepth <= max_depth:
		if left_neg != 0 and left_pos != 0:
			root.leftchildnode = Node()
			root.leftchildnode = create_node(root.leftchildnode, treedepth+1, left_array,left_pos,left_neg, accur, size_x)
		else:
			root.leftchildnode = Node()
			root.leftchildnode.label = setlabel(left_pos, left_neg)
		
		if right_neg != 0 and right_pos != 0:
			root.rightchildnode = Node()
			root.rightchildnode = create_node(root.rightchildnode, treedepth+1, right_array,right_pos,right_neg, accur, size_x)
		else:
			root.rightchildnode = Node()
			root.rightchildnode.label = setlabel(right_pos, right_neg)

	return root


def sort_cut(total_train, nodefeature, val):
	x_array_sorted = total_train[total_train[:,nodefeature].argsort()]
	x_array_sorted_t = np.transpose(x_array_sorted)

	split_point = (x_array_sorted_t[nodefeature] < val).sum()

	left_array = x_array_sorted[0:split_point]
	right_array = x_array_sorted[split_point:]

	count_left_neg,count_left_pos,count_right_neg, count_right_pos = split_data(left_array, right_array)
	
	return left_array, right_array, count_left_neg,count_left_pos,count_right_neg, count_right_pos



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

def Cal_Error(data, root, pred_y):

	data_y = (data.T)[0]
	add = predict_y + data_y
	result = (add==0).sum()
	total_num = len(data)
	err = result/total_num
	acc = 1-err
	return err, acc

def alpha(data, root, pred_y):
	error, accur = Cal_Error(data, root,pred_y)
	return ((1/2)*math.log((1-error)/error))

def ChangeDistribution(data, root,pred_y):

	data_y = (data.T)[0]
	alp = alpha(data, root,pred_y)
	D_value = 0
	D_1 = []
	for i in range(0, len(D)):
		if pred_y[i] != data_y[i]:
			D_value = D[i] * math.exp(alp)
			D_1.append(D_value)
		else:
			D_value = D[i] * math.exp(-alp)
			D_1.append(D_value)
	
	next_D = np.array(D_1)
	return next_D

def accu_calculate(pred_y,y_array):
	calculate_y = np.add(pred_y,y_array)
	error_value = (calculate_y == 0).sum()
	accu = (pred_y.shape[0]-error_value)/pred_y.shape[0]
	return accu



y_array_train = yvalues('pa3_train_reduced.csv')
x_array_train = xvalues('pa3_train_reduced.csv')
total_train = read_data('pa3_train_reduced.csv')

y_array_valid = yvalues('pa3_valid_reduced.csv')
x_array_valid = xvalues('pa3_valid_reduced.csv')
total_valid = read_data('pa3_valid_reduced.csv')

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


max_depth = 8
D = np.repeat(1/4888, 4888)
size_x = total_train.shape[1]
result_t, result_v = np.zeros(total_train.shape[0]), np.zeros(total_valid.shape[0])
tmp_y = np.zeros(y_array_train.shape[0])




######################################

tree_list = list()
for i in range(1):
	
	tree_list.append(DecisionTree())
	accur = dict()
	pos_idx = np.where(y_array_train==1)[0] 
	root_pos = np.sum(D[pos_idx])
	neg_idx = np.where(y_array_train==-1)[0]
	root_neg = np.sum(D[neg_idx])
	create_node(tree_list[i].root, 0, total_train, root_pos, root_neg, accur,size_x-1)


	predict_y = output_pred_y(total_train, tree_list[i].root)
	tmp_y = predict_y
	predict_y_v = output_pred_y(total_valid,tree_list[i].root)
	alp = alpha(total_train, tree_list[i].root, predict_y)
	result_v = np.add(alp*predict_y_v, result_v)
	result_t = np.add(alp*predict_y, result_t)
	D = np.copy(ChangeDistribution(total_train, tree_list[i].root, predict_y))
	

total_result = np.sign(result_t)
total_result_v = np.sign(result_v)
for k in range(total_result.shape[0]):
	if total_result[k] == 0:
		total_result[k] == rd.choice([1,-1])

for k in range(total_result_v.shape[0]):
	if total_result_v[k] == 0:
		total_result_v[k] == rd.choice([1,-1])
print("L=", i+1)
print("train:",accu_calculate(total_result, y_array_train))
print("valid:",accu_calculate(total_result_v, y_array_valid))






tree_list = list()
for i in range(5):
	
	tree_list.append(DecisionTree())
	accur = dict()
	pos_idx = np.where(y_array_train==1)[0] 
	root_pos = np.sum(D[pos_idx])
	neg_idx = np.where(y_array_train==-1)[0]
	root_neg = np.sum(D[neg_idx])
	create_node(tree_list[i].root, 0, total_train, root_pos, root_neg, accur,size_x-1)


	predict_y = output_pred_y(total_train, tree_list[i].root)
	tmp_y = predict_y
	predict_y_v = output_pred_y(total_valid,tree_list[i].root)
	alp = alpha(total_train, tree_list[i].root, predict_y)
	result_v = np.add(alp*predict_y_v, result_v)
	result_t = np.add(alp*predict_y, result_t)
	D = np.copy(ChangeDistribution(total_train, tree_list[i].root, predict_y))
	

total_result = np.sign(result_t)
total_result_v = np.sign(result_v)
for k in range(total_result.shape[0]):
	if total_result[k] == 0:
		total_result[k] == rd.choice([1,-1])

for k in range(total_result_v.shape[0]):
	if total_result_v[k] == 0:
		total_result_v[k] == rd.choice([1,-1])
print("L=", 5)
print("train:",accu_calculate(total_result, y_array_train))
print("valid:",accu_calculate(total_result_v, y_array_valid))



tree_list = list()
for i in range(10):
	
	tree_list.append(DecisionTree())
	accur = dict()
	pos_idx = np.where(y_array_train==1)[0] 
	root_pos = np.sum(D[pos_idx])
	neg_idx = np.where(y_array_train==-1)[0]
	root_neg = np.sum(D[neg_idx])
	create_node(tree_list[i].root, 0, total_train, root_pos, root_neg, accur,size_x-1)


	predict_y = output_pred_y(total_train, tree_list[i].root)
	tmp_y = predict_y
	predict_y_v = output_pred_y(total_valid,tree_list[i].root)
	alp = alpha(total_train, tree_list[i].root, predict_y)
	result_v = np.add(alp*predict_y_v, result_v)
	result_t = np.add(alp*predict_y, result_t)
	D = np.copy(ChangeDistribution(total_train, tree_list[i].root, predict_y))
	

total_result = np.sign(result_t)
total_result_v = np.sign(result_v)
for k in range(total_result.shape[0]):
	if total_result[k] == 0:
		total_result[k] == rd.choice([1,-1])

for k in range(total_result_v.shape[0]):
	if total_result_v[k] == 0:
		total_result_v[k] == rd.choice([1,-1])
print("L=", 10)
print("train:",accu_calculate(total_result, y_array_train))
print("valid:",accu_calculate(total_result_v, y_array_valid))

tree_list = list()
for i in range(20):
	
	tree_list.append(DecisionTree())
	accur = dict()
	pos_idx = np.where(y_array_train==1)[0] 
	root_pos = np.sum(D[pos_idx])
	neg_idx = np.where(y_array_train==-1)[0]
	root_neg = np.sum(D[neg_idx])
	create_node(tree_list[i].root, 0, total_train, root_pos, root_neg, accur,size_x-1)


	predict_y = output_pred_y(total_train, tree_list[i].root)
	tmp_y = predict_y
	predict_y_v = output_pred_y(total_valid,tree_list[i].root)
	alp = alpha(total_train, tree_list[i].root, predict_y)
	result_v = np.add(alp*predict_y_v, result_v)
	result_t = np.add(alp*predict_y, result_t)
	D = np.copy(ChangeDistribution(total_train, tree_list[i].root, predict_y))
	

total_result = np.sign(result_t)
total_result_v = np.sign(result_v)
for k in range(total_result.shape[0]):
	if total_result[k] == 0:
		total_result[k] == rd.choice([1,-1])

for k in range(total_result_v.shape[0]):
	if total_result_v[k] == 0:
		total_result_v[k] == rd.choice([1,-1])
print("L=", 20)
print("train:",accu_calculate(total_result, y_array_train))
print("valid:",accu_calculate(total_result_v, y_array_valid))