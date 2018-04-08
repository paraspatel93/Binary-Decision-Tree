# -*- coding: utf-8 -*-
"""
Decision Tree Impleentation.
Created on Fri Feb 02 17:49:39 2018

@author: Paras Patel
"""
import numpy as np
import csv
import sys

depth_desired =int(sys.argv[3])
depth_count=0

def unique(col):
    label=(np.unique(col))
    if len(label)==1:
        val=0
        for row in range(0,len(col)):
            if label[0]==col[row]:
                val=val+1
        return [label,[val]]
    else:
        val=[0,0]
        for row in range(0,len(col)):
            if label[0]==col[row]:
                val[0]=val[0]+1
            else: val[1]=val[1]+1
        return [label,np.array(val)]
    
#Returns Entropy of given data.
def entropy(data):
    ent=0
    label=(unique(data[:,-1]))
    totalcount=np.sum(label[1])
    for i in range(0,len(label[1])):
        p=float(label[1][i])/totalcount
        ent += p*np.log2(1/p) 
    return ent

#Returns unique value of specific column.
def unique_val(rows,col):
    return unique(rows.T[col])

#Checks if the given data is numeric.
def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

class Attribute:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        return val == self.value
    def __repr__(self):
        return "%s" % (
            header[self.column])

#This function partitions the data into true/false or Yes/No.(Binary Split)
def data_partition(rows,attribute):
    true_rows, false_rows = [], []
    for row in rows:
        if attribute.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

#this function calculates the information gain at the node.
def info_gain(left_node,right_node,entr):
    p = float(len(left_node))/(len(left_node)+len(right_node))
    infogain = entr+(float(-p*entropy(np.array(left_node)))-((1-p)*entropy(np.array(right_node))))
    return float(infogain)

#Finds the attribute to split where information gain is the highest.
def find_best_split(rows):
    max_gain = 0  
    best_attribute = None  
    ent = entropy(rows)
    n_features = len(rows[0]) - 1 
 
    for col in range(n_features): 
        values=np.unique(rows)
        for val in values:  
            attribute = Attribute(col, val)
            true_rows, false_rows = data_partition(rows, attribute)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, ent)

            if gain >= max_gain:
                max_gain, best_attribute = gain, attribute
    return max_gain, best_attribute

#Leaf of Decision Tree and predictions stores unique label with its count.
class Leaf:
    def __init__(self,rows):
        self.predictions=unique(np.transpose(rows)[-1])

#This Class stores Node information with unique labels and its count.
class Decision_Node:
    def __init__(self,
                 attribute,
                 true_branch,
                 false_branch,tr_count,fl_count):
        self.attribute = attribute
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.trcount=tr_count
        self.flcount=fl_count

#This function build the tree by training the data for maximum depth.
def build_tree(rows):
    global depth_count
    global depth_desired 
    
    gain, attribute = find_best_split(rows)
    
    if gain == 0:
        return Leaf(rows)
    
    
    true_rows, false_rows = data_partition(rows, attribute)
    true_branch = build_tree(np.array(true_rows))
    false_branch = build_tree(np.array(false_rows))
    return Decision_Node(attribute, true_branch, false_branch,unique(np.transpose(true_rows)[-1]),unique(np.transpose(false_rows)[-1]))
    
#This function prints the tree.   
def print_tree(node, spacing="| "):
    global depth_count
    if isinstance(node, Leaf):
        return
    print ("{0}{1}".format(spacing , str(node.attribute)))
    print ("{0}{1} {3} /{2}".format(spacing , ' | y:',node.trcount[0],node.trcount[1]))
    print_tree(node.true_branch, spacing + " | ")
    print ("{0}{1} {3} /{2} ".format(spacing , ' | n:',node.flcount[0],node.flcount[1]))
    print_tree(node.false_branch, spacing + " | ")

#This functions classifies the given data.
def classify(row, node):
    global depth_count
    if isinstance(node, Leaf):
        return node.predictions
    if(depth_count<depth_desired-1):
        depth_count=depth_count+1
        if node.attribute.match(row):
            return classify(row, node.true_branch)
        else:
            return classify(row, node.false_branch)
    else:
        if node.attribute.match(row):
            return node.trcount
        else:
            return node.flcount
        
#This function calculates the error
def error(real_data,predict_data):
    false=0
    for i in range(0,len(predict_data)):
        if (real_data[i]!=predict_data[i]):
            false=false+1
    return float(false)/len(predict_data)
    
#Open all files.
file_train=open(sys.argv[1],"r")
file_test=open(sys.argv[2],"r")
train_out=open(sys.argv[4],"w")
test_out=open(sys.argv[5],"w")
metric_out=open(sys.argv[6],"w")


train_data = np.array(list(csv.reader(file_train, delimiter=',')))
test_data = np.array(list(csv.reader(file_test, delimiter=',')))
header=train_data[0]
train_data=np.delete(train_data,0,0)
test_data=np.delete(test_data,0,0)
my_tree=build_tree(train_data)
temp1,temp2=[],[]

if depth_desired==0:  #if depth desired is 0, it is a majority vote classifier.
    for i in range(0,len(train_data)):
        majority=unique(np.transpose(train_data)[-1])
        train_predict=majority[0][np.argmax(majority[1])]
        temp1.append(train_predict)
        train_out.write("%s\n" % train_predict)
    for i in range(0,len(test_data)):
        majority=unique(np.transpose(train_data)[-1])
        test_predict=majority[0][np.argmax(majority[1])]
        temp2.append(test_predict)
        test_out.write("%s\n" % test_predict)
else:
    for i in range(0,len(train_data)):
        train_predict=classify(train_data[i],my_tree) 
        train_predict=train_predict[0][np.argmax(train_predict[1])]  
        temp1.append(train_predict)
        train_out.write("%s\n" % train_predict)
        
    for i in range(0,len(test_data)):
        test_predict=classify(test_data[i],my_tree)
        test_predict=test_predict[0][np.argmax(test_predict[1])]
        temp2.append(test_predict)
        test_out.write("%s\n" % test_predict)
    
print_tree(my_tree)
err1=error(train_data.T[-1],temp1)    #Training Error
err2=error(test_data.T[-1],temp2)     #Test Error
metric_out.write("error(train): %f\n" % err1)
metric_out.write("error(test): %f" % err2)

file_train.close()
file_test.close()
test_out.close()
train_out.close()
metric_out.close()

#END 