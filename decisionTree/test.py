#coding=utf-8
__author__ = 'baconLIN'

import trees

myDat,labels=trees.createDataSet()
print(myDat)
shannonEnt = trees.calcShannonEnt(myDat)
print(shannonEnt)

mySplit1 = trees.splitDataSet(myDat,0,1)
print mySplit1
shannonEntSplit1 = trees.calcShannonEnt(mySplit1)
print(shannonEntSplit1)
mySplit2 = trees.splitDataSet(myDat,0,0)
print mySplit2
shannonEntSplit2 = trees.calcShannonEnt(mySplit2)
print(shannonEntSplit2)
mySplit3 = trees.splitDataSet(myDat,1,1)
print mySplit3
shannonEntSplit3 = trees.calcShannonEnt(mySplit3)
print(shannonEntSplit3)

bestFeature = trees.chooseBestFeatureToSplit(myDat)
print(bestFeature)

myTree = trees.createTree(myDat,labels)
print(myTree)

import treePlotter
myTree2 = treePlotter.retrieveTree(0)
treePlotter.createPlot(myTree2)

trees.storeTree(myTree2,'classifierStorage2.txt')
print trees.grabTree('classifierStorage2.txt')

fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree=trees.createTree(lenses,lensesLabels)
print lensesTree
treePlotter.createPlot(lensesTree)