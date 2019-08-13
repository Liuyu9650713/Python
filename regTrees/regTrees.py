# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 09:50:02 2019

@author: Administrator
"""

from numpy import *
import matplotlib.pyplot as plt
 
def regTrees_prune():
    data=loadDataSet('ex2.txt')
    myMat=mat(data)
    a=createTree(myMat)
    print(a)
    data=loadDataSet('ex2test.txt')
    testData=mat(data)
    b=prune(a,testData)
    print(b)

def regTrees_main():
    myDat = loadDataSet('ex00.txt')
    myMat = mat(myDat)
    retTree = createTree(myMat)
    print(retTree)
    myDat1 = loadDataSet('ex0.txt')
    myMat1 = mat(myDat1)
    retTree1 = createTree(myMat1)
    print(retTree1)

def showDataSet():
    myDat=loadDataSet('ex00.txt')
    myMat=mat(myDat)
    fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(20,6))
    axs[0].scatter(myMat[:,0].flatten().A[0],myMat[:,1].flatten().A[0], c = 'blue')
    myDat1=loadDataSet('ex0.txt')
    myMat1=mat(myDat1)
    axs[1].scatter(myMat1[:,1].flatten().A[0],myMat1[:,2].flatten().A[0], c = 'blue')
    plt.show()

def showDataSet1():
    myDat=loadDataSet('ex2.txt')
    myMat=mat(myDat)
    fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(20,6))
    axs[0].scatter(myMat[:,0].flatten().A[0],myMat[:,1].flatten().A[0], c = 'blue')
    myDat1=loadDataSet('ex2test.txt')
    myMat1=mat(myDat1)
    axs[1].scatter(myMat1[:,0].flatten().A[0],myMat1[:,1].flatten().A[0], c = 'blue')
    plt.show()

#作用：从文件导入数据
#输入：文件名
#输出：数据数组
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    ##依次读取每行
    for line in fr.readlines():
        #去掉每行头尾空白,并以制表符进行分割返回列表
        curLine = line. strip().split('\t')
        #将每行映射成浮点数
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat
# 作用：从文件导入数据
# 输入：数据矩阵，待切分特征值，阈值
# 输出：切分后的数据集
def binSplitDataSet(dataSet, feature, value):
    # 书中最后有[0]，练习发现只会返回1*n矩阵，因此删掉
    # nonzero()返回的是列表中值不为零的元素的下标
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0,mat1
# 作用：目标变量的均值
# 输入：数据集
# 输出：目标变量的均值
def regLeaf(dataSet):
    return mean(dataSet[:, -1])
 
# 作用：目标变量的总方差
# 输入：数据集
# 输出：目标变量的总方差
def regErr(dataSet):
    #数据集最后一列的均方差乘以数据集中样本的个数，得总方差
    return var(dataSet[:, -1]) * shape(dataSet)[0]
 
# 作用：找到数据的最佳二元切分方式
# 输入：数据集(矩阵)，建立叶节点的函数，误差计算函数，包含树构建所需其他参数的元组
# 输出：最佳切分特征和特征值
def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
    tolS = ops[0]#容许的误差下降值
    tolN = ops[1]#切分的最少样本数
    #如果所有值都相等则退出
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)#数据集的行数和列数
    S = errType(dataSet)#数据集的总方差
    bestS = inf#切分点求得两边数据的方差，统计学习方法书上(5.21)求得的最小值
    bestIndex = 0#最佳切分特征
    bestValue = 0#最佳切分特征值
    for featIndex in range(n-1):
        #书中代码有错，需改成如下形式，转置后转换为列表，对每个特征的特征值
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 有切分的数据集太小，跳过该种切分方式
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            #切分后的总方差
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #如果误差减少的不大则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    #如果切分出的数据集很小则退出，预防进入两个for循环之后并未更新bestIndex, bestValue的情况
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue    
#作用：创建树
#输入：数据集（矩阵），建立叶节点的函数，误差计算函数，包含树构建所需其他参数的元组
#输出：树
def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
    #寻找最佳的切分特征和特征值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    #定义一个空字典
    retTree = {}
    #将最优特征和特征值存入字典
    retTree['spInd0'] = feat
    retTree['spVal1'] = val
    #将数据切分成左右两份
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    #递归调用createTree函数
    retTree['left2'] = createTree(lSet, leafType, errType, ops)
    retTree['right3'] = createTree(rSet, leafType, errType, ops)
    return retTree

#作用：测试输入变量是否是一棵树
#输入：输入变量
#输出：布尔类型的结果，是一棵树则返回True
def isTree(obj):
    #如果是一棵树，则类型为'dict'即字典
    return (type(obj).__name__ == 'dict')
 
#作用：得到树的平均值
#输入：树
#输出：树的平均值
def getMean(tree):
    if isTree(tree['right3']):
        tree['right3'] = getMean(tree['right3'])
    if isTree(tree['left2']):
        tree['left2'] = getMean(tree['left2'])
    #如果左子树和右子树的样例个数不相等，意义是什么？
    return (tree['left2'] + tree['right3']) / 2.0
 
#作用：剪枝
#输入：待剪枝的树，剪枝所需的测试数据
#输出：剪好的树
def prune(tree, testData):
    if shape(testData)[0] == 0:#确认测试集是否为空
        return getMean(tree)
    #如果两个分支有一个是子树，则对测试数据进行划分
    if (isTree(tree['right3']) or isTree(tree['left2'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd0'], tree['spVal1'])
    # 对左子树进行剪枝
    if isTree(tree['left2']):
        tree['left2'] = prune(tree['left2'], lSet)
    # 对右子树进行剪枝
    if isTree(tree['right3']):
        tree['right3'] = prune(tree['right3'], rSet)
    #两个分支都不是子树
    if not isTree(tree['left2']) and not isTree(tree['right3']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd0'], tree['spVal1'])
        #没有合并的误差
        errorNoMerge = sum(power(lSet[:, -1] - tree['left2'], 2))+sum(power(rSet[:, -1] - tree['right3'], 2))
        #计算树的平均数
        treeMean = (tree['left2'] + tree['right3']) / 2.0
        #合并后的误差
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


#生成决策树
#regTrees_main()
#showDataSet()

#剪枝
#showDataSet1()
#regTrees_prune()