__author__ = 'baconLIN'
import kNN
from numpy import *
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
print datingDataMat
print datingLabels[0:20]

import matplotlib
import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
# plt.show()

normMat,ranges,minVals=kNN.autoNorm(datingDataMat)
print normMat
print ranges
print minVals

# kNN.datingClassTest('datingTestSet2.txt')
# kNN.classifyPerson()

# testVector = kNN.img2vector('testDigits/0_26.txt')
# print testVector[0,0:31]
# print testVector[0,32:63]
kNN.handwritingClassTest()