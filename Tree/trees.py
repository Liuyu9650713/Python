# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 08:54:39 2019

@author: Administrator
"""

from math import log  
import operator  

# 计算信息熵  
def CalcShannonEnt(dataSet):  
    #计算数据集的输入个数  
    numEntries = len(dataSet)  
    #[]列表,{}元字典,()元组  
    # 创建存储标签的元字典  
    labelCounts = {}  
    #对数据集dataSet中的每一行featVec进行循环遍历  
    for featVec in dataSet:  
        # currentLabels为featVec的最后一个元素即数据的类别  
        currentLabels =featVec[-1]  
        # 如果标签currentLabels不在元字典对应的key中  
        if currentLabels not in labelCounts.keys():  
            # 将标签currentLabels放到字典中作为key，并将值赋为0  
            labelCounts[currentLabels] = 0  
        # 将currentLabels对应的值加1  
        labelCounts[currentLabels] += 1  
    # 定义香农熵shannonEnt  
    shannonEnt = 0.0  
    # 遍历元字典labelCounts中的key，即标签  
    for key in labelCounts:  
        # 计算每一个标签出现的频率，即概率  
        prob = float(labelCounts[key])/numEntries  
        # 根据信息熵公式计算每个标签信息熵并累加到shannonEnt上  
        shannonEnt -= prob*log(prob,2)  
    # 返回求得的整个标签对应的信息熵  
    return shannonEnt  


# 创建数据集  
def createDataSet():  
    dataSet = [[1,1,'yes'],  
               [1,1,'yes'],  
               [1,0,'no'],  
               [0,1,'no'],  
               [0,1,'no']]  
    labels=['no surfacing','flippers']  
    return dataSet,labels  


# 分割数据集  
# dataSet数据集，axis是对应的要分割数据的特征，value是要分割的列按哪个值分割，即找到含有该值的数据  
def splitDataSet(dataSet,axis,value):  
    # 定义要返回的数据集  
    retDataSet = []  
    # 遍历数据集中的每个特征，即输入数据  
    for featVec in dataSet:  
        # 如果列标签对应的值为value，则将该条(行)数据加入到retDataSet中  
        if featVec[axis] == value:  
            # 取featVec的0-axis个数据，不包括axis，放到reducedFeatVec中  
            reducedFeatVec = featVec[:axis]  
            # 取featVec的axis+1到最后的数据，放到reducedFeatVec的后面  
            reducedFeatVec.extend(featVec[axis+1:])  
            # 将reducedFeatVec添加到分割后的数据集retDataSet中，同时reducedFeatVec，retDataSet中没有了axis列的数据  
            retDataSet.append(reducedFeatVec)  
    # 返回分割后的数据集  
    return retDataSet  


# 选择使分割后信息增益最大的特征，即对应的列  
def chooseBestFeatureToSplit(dataSet):  
    # 获取特征的数目，从0开始，dataSet[0]是一条数据  
    numFeatures = len(dataSet[0]) - 1  
    # 计算数据集当前的信息熵  
    baseEntropy = CalcShannonEnt(dataSet)  
    # 定义最大的信息增益  
    bestInfoGain = 0.0  
    # 定义分割后信息增益最大的特征  
    bestFeature = -1  
    # 遍历特征，即所有的列，计算每一列分割后的信息增益，找出信息增益最大的列  
    for i in range(numFeatures):  
        # 取出第i列特征赋给featList  
        featList = [example[i] for example in dataSet]  
        # 去掉重复项，使得特征列的数据具有唯一性  
        uniqueVals = set(featList)  
        # 定义分割后的信息熵  
        newEntropy = 0.0  
        # 遍历特征列的所有值(值是唯一的，重复值已经合并)，分割并计算信息增益  
        for value in uniqueVals:  
            # 按照特征列的每个值进行数据集分割  
            subDataSet = splitDataSet(dataSet, i, value)   
            # 计算分割后的每个子集的概率(频率)  
            prob = len(subDataSet) / float(len(dataSet))  
            # 计算分割后的子集的信息熵并相加，得到分割后的整个数据集的信息熵  
            newEntropy +=prob * CalcShannonEnt(subDataSet)  
        # 计算分割后的信息增益  
        infoGain = baseEntropy - newEntropy  
        # 如果分割后信息增益大于最好的信息增益  
        if(infoGain > bestInfoGain):  
            # 将当前的分割的信息增益赋值为最好信息增益  
            bestInfoGain = infoGain  
            # 分割的最好特征列赋为i  
            bestFeature = i  
    # 返回分割后信息增益最大的特征列  
    return bestFeature  


# 对类标签进行投票 ，找标签数目最多的标签  
def majorityCnt(classList):  
    # 定义标签元字典，key为标签，value为标签的数目  
    classCount = {}  
    # 遍历所有标签  
    for vote in classList:  
        #如果标签不在元字典对应的key中  
        if vote not in classCount.keys():  
            # 将标签放到字典中作为key，并将值赋为0  
            classCount[vote] = 0  
        # 对应标签的数目加1  
        classCount[vote] += 1  
    # 对所有标签按数目排序  
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)  
    # 返回数目最多的标签  
    return sortedClassCount[0][0]  


# 创建决策树  
def createTree(dataSet,labels):  
    # 将dataSet的最后一列数据(标签)取出赋给classList，classList存储的是标签列  
    classList = [example[-1] for example in dataSet]  
    # 判断是否所有的列的标签都一致  
    if classList.count(classList[0]) == len(classList):  
        # 直接返回标签列的第一个数据  
        return classList[0]  
    # 判断dataSet是否只有一条数据  
    if len(dataSet[0]) == 1:  
        # 返回标签列数据最多的标签  
        return majorityCnt(classList)  
    # 选择一个使数据集分割后最大的特征列的索引  
    bestFeat = chooseBestFeatureToSplit(dataSet)  
    # 找到最好的标签  
    bestFeatLabel = labels[bestFeat]  
    # 定义决策树，key为bestFeatLabel，value为空  
    myTree = {bestFeatLabel:{}}  
    # 删除labels[bestFeat]对应的元素  
    del(labels[bestFeat])  
    # 取出dataSet中bestFeat列的所有值  
    featValues = [example[bestFeat] for example in dataSet]  
    # 将特征对应的值放到一个集合中，使得特征列的数据具有唯一性  
    uniqueVals = set(featValues)  
    # 遍历uniqueVals中的值  
    for value in uniqueVals:  
        # 子标签subLabels为labels删除bestFeat标签后剩余的标签  
        subLabels = labels[:]  
        # myTree为key为bestFeatLabel时的决策树  
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat, value), subLabels)  
    # 返回决策树  
    return myTree  


# 决策树分类函数  
def classify(inputTree,featLabels,testVec):  
    # 得到树中的第一个特征  
    firstStr = list(inputTree.keys())[0]  
    # 得到第一个对应的值  
    secondDict = inputTree[firstStr]  
    # 得到树中第一个特征对应的索引  
    featIndex = featLabels.index(firstStr)  
    # 遍历树  
    for key in secondDict.keys():  
        # 如果在secondDict[key]中找到testVec[featIndex]  
        if testVec[featIndex] == key:  
            # 判断secondDict[key]是否为字典  
            if type(secondDict[key]).__name__ == 'dict':  
                # 若为字典，递归的寻找testVec  
                classLabel = classify(secondDict[key], featLabels, testVec)  
            else:  
                # 若secondDict[key]为标签值，则将secondDict[key]赋给classLabel  
                classLabel = secondDict[key]  
    # 返回类标签  
    return classLabel  


# 将获取到的决策树数据保存到文件中去 
def storeTree(inputTree,filename):  
    # 导入pyton模块  
    import pickle  
    # 以写的方式打开文件  
    fw = open(filename,'w')  
    # 决策树序列化  
    pickle.dump(inputTree,fw)
    #关掉文件
    fw.close()          
# 读取文件中的决策树        
def grabTree(filename):  
    import pickle  
    fr = open(filename)  
    # 返回读到的树  
    return pickle.load(fr)  