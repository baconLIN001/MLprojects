# coding=utf-8
__author__ = 'bacon'

import regTrees
from numpy import *

testMat = mat(eye(4))
print testMat

mat0,mat1 = regTrees.binSplitDataSet(testMat,1,0.5)
print mat0
print mat1

# myDat = regTrees.loadDataSet('ex00.txt')
# myMat = mat(myDat)
# print regTrees.createTree(myMat)
#
# myDat2=regTrees.loadDataSet('ex0.txt')
# myMat2=mat(myDat2)
# print regTrees.createTree(myMat2)
#
# print regTrees.createTree(myMat,ops=(0,1))
#
myDat3 = regTrees.loadDataSet('ex2.txt')
myMat3 = mat(myDat3)
# print regTrees.createTree(myMat3)

myTree=regTrees.createTree(myMat3,ops=(0,1))
myDatTest=regTrees.loadDataSet('ex2test.txt')
myMatTest=mat(myDatTest)
print regTrees.prune(myTree,myDatTest)