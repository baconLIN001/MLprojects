# coding=utf-8
__author__ = 'bacon'

from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fileread = open(fileName)
    for line in fileread.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat
#通过数组过滤方式将数据集切分成两个子集返回
"""
    :param:dataSet:数据集
    :param:feature:待切分特征
    :param:value:该特征的某个值，切分点
"""
def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[nonzero(dataSet[:,feature]>value)[0],:][0]
    mat1 = dataSet[nonzero(dataSet[:,feature]<=value)[0],:][0]
    return mat0,mat1

#生成叶节点，回归树中，为均值
def regLeaf(dataSet):
    return mean(dataSet[:,-1])
#计算目标变量的总方差，用均方差乘以数据集中样本的个数
def regErr(dataSet):
    return var(dataSet[:,-1])*shape(dataSet)[0]

"""
    :param:dataSet:数据集
    :param:leafType:建立叶节点的函数
    :param:errType:误差计算函数
    :param:ops:一个包含树构建所需其他参数的元组
"""
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None: return val
    retTree = {}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    retTree['left']=createTree(lSet,leafType,errType,ops)
    retTree['right']=createTree(rSet.leafType,errType,ops)
    return retTree
#找到数据的最佳二元划分方式
"""
    :param:dataSet:数据集
    :param:leafType:建立叶节点的函数
    :param:errType:误差计算函数
    :param:ops:一个包含树构建所需其他参数的元组
"""
def chooseBestSplit(dataSet,leafType=regLeaf, errType=regErr,ops=(1,4)):
    #用户指定的参数，用于控制函数的停止时机
    tolS = ops[0] #容许的误差下降值
    tolN = ops[1] #切分的最小样本数
    #如果所有值相等则退出（统计不同剩余特征值的数目，为1返回）
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:
        return None,leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex]):
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if (shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN): continue
            newS = errType(mat0)+errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue= splitVal
                bestS = newS
    #如果切分后数据集效果提升不大（小于给定值），不切分直接创建叶节点
    if(S-bestS)<tolS:
        return None,leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    #子集小于设定，停止切分
    if(shape(mat0)[0]<tolN)or(shape(mat1)[0]<tolN):
        return None,leafType(dataSet)
    return bestIndex,bestValue