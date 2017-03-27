# coding=utf-8
__author__ = 'bacon'

import regTrees
from numpy import *

testMat = mat(eye(4))
print testMat

mat0,mat1 = regTrees.binSplitDataSet(testMat,1,0.5)
print mat0
print mat1