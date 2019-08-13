# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 08:50:38 2019
@author: 刘誉
"""
from numpy import *
import operator
from math import log 
import matplotlib.pyplot  as plt 
import matplotlib

font = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')

def ID3_main():
    data=loadDataSet('shuju1.txt')
    dataSet,labels=translate(data)
    tree=createTree(dataSet,labels)
    createPlot(tree)
"""
得到决策树的数据
"""
#作用：从文件导入数据
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    ##依次读取每行
    for line in fr.readlines():
        #去掉每行头尾空白,并以制表符进行分割返回列表
        curLine = line. strip().split()
        dataMat.append(curLine)
    return dataMat
#将数据分类处理
def translate(Data):
    mat0=[x[1:] for x in Data[1:]]
    #mat1取所有的特征标签
    mat1=Data[0][1:-1]
    return mat0,mat1
"""
构造决策树
"""
# 创建决策树  
def createTree(dataSet,labels):  
    # 将dataSet的最后一列数据(标签)取出赋给classList，classList存储的是标签列
    #for example in dataSet是一行一行读取数据的
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
        # myTree为key为bestFeatLabel时的决策树 ，myTree[bestFeatLabel][value]是一个嵌套字典所以是二维的，两个都是键
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat, value), subLabels)  
    # 返回决策树  
    return myTree  
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
    # 获取特征的数目，从0开始，dataSet[0]是一行数据  去掉最后一列的类
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
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)  
    # 返回数目最多的标签  
    return sortedClassCount[0][0] 

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

"""画出树形结构的决策树"""
def createPlot(inTree):  
    # 定义一块画布(画布是自己的理解)  
    fig = plt.figure(1,facecolor='white')  
    # 清空画布  
    fig.clf()  
    # 定义横纵坐标轴，无内容  
    axprops = dict(xticks=[],yticks=[])  
    # 绘制图像，无边框，无坐标轴  
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)  
    # plotTree.totalW保存的是树的宽  
    plotTree.totalW = float(getNumLeafs(inTree))  
    # plotTree.totalD保存的是树的高  
    plotTree.totalD = float(getTreeDepth(inTree))  
    # 决策树起始横坐标  
    plotTree.xOff = - 0.5 / plotTree.totalW #从0开始会偏右  
#    print  plotTree.xOff  
    # 决策树的起始纵坐标  
    plotTree.yOff = 1.0  
    # 绘制决策树  
    plotTree(inTree,(0.5,1.0),'')  
    # 显示图像  
    plt.show()  

# 定义决策树决策结果的属性，用字典来定义  
# 下面的字典定义也可写作 decisionNode={boxstyle:'sawtooth',fc:'0.8'}  
# boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细  
decisionNode = dict(boxstyle="sawtooth",fc="0.8")  
# 定义决策树的叶子结点的描述属性  
leafNode = dict(boxstyle="round4",fc="0.8")  
# 定义决策树的箭头属性  
arrow_args = dict(arrowstyle="<-")  

# 绘制结点  
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
      #annotate是关于一个数据点的文本  
      #nodeTxt为要显示的文本，centerPt为文本的中心点，箭头所在的点，parentPt为指向文本的点       
    createPlot.ax1.annotate(nodeTxt,fontproperties = font,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction', 
                            va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)  
       
# 获得决策树的叶子结点数目  
def getNumLeafs(myTree):  
    # 定义叶子结点数目  
    numLeafs = 0  
    # 获得myTree的第一个键值，即第一个特征，分割的标签  
    firstStr = list(myTree.keys())[0]  
    # 根据键值得到对应的值，即根据第一个特征分类的结果  
    secondDict = myTree[firstStr]  
    # 遍历得到的secondDict  
    for key in list(secondDict.keys()):  
        # 如果secondDict[key]为一个字典，即决策树结点  
        if type(secondDict[key]).__name__ == 'dict':  
            # 则递归的计算secondDict中的叶子结点数，并加到numLeafs上  
            numLeafs += getNumLeafs(secondDict[key])  
        # 如果secondDict[key]为叶子结点  
        else:  
            # 则将叶子结点数加1      
            numLeafs += 1  
    # 返回求的叶子结点数目  
    return numLeafs  

# 获得决策树的深度  
def getTreeDepth(myTree):  
    # 定义树的深度  
    maxDepth = 0  
    # 获得myTree的第一个键值，即第一个特征，分割的标签  
    firstStr = list(myTree.keys())[0]  
    # 根据键值得到对应的值，即根据第一个特征分类的结果  
    secondDict = myTree[firstStr]  
    for key in secondDict.keys():  
        # 如果secondDict[key]为一个字典  
        if type(secondDict[key]).__name__ == 'dict':  
            # 则当前树的深度等于1加上secondDict的深度，只有当前点为决策树点深度才会加1  
            thisDepth = 1 + getTreeDepth(secondDict[key])  
            # 如果secondDict[key]为叶子结点  
        else:  
            # 则将当前树的深度设为1      
            thisDepth = 1  
    # 如果当前树的深度比最大数的深度  
        if thisDepth > maxDepth:  
            maxDepth = thisDepth  
    # 返回树的深度  
    return maxDepth

# 绘制中间文本
def plotMidText(cntrPt,parentPt,txtString):
    # 求中间点的横坐标
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    # 求中间点的纵坐标
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    # 绘制树结点
    createPlot.ax1.text(xMid,yMid,txtString,fontproperties = font)

# 绘制决策树  
def plotTree(myTree,parentPt,nodeTxt):  
    # 定义并获得决策树的叶子结点数  
    numLeafs = getNumLeafs(myTree)  
    #得到决策树的高度   
    depth=getTreeDepth(myTree)  
    # 得到第一个特征  
    firstStr = list(myTree.keys())[0]
    # 计算坐标，x坐标为当前树的叶子结点数目除以整个树的叶子结点数再除以2，y为起点  
    cntrPt = (plotTree.xOff + (1.0 +float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)  
    # 绘制中间结点，即决策树结点，也是当前树的根结点，这句话没感觉出有用来，注释掉照样建立决策树，理解浅陋了，理解错了这句话的意思，下面有说明
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 绘制决策树结点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  
    # 根据firstStr找到对应的值  
    secondDict = myTree[firstStr]  
    # 因为进入了下一层，所以y的坐标要变 ，图像坐标是从左上角为原点  
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD  
    # 遍历secondDict  
    for key in secondDict.keys():  
        # 如果secondDict[key]为一棵子决策树，即字典  
        if type(secondDict[key]).__name__ == 'dict':  
            # 递归的绘制决策树  
            plotTree(secondDict[key],cntrPt,str(key))  
        # 若secondDict[key]为叶子结点  
        else:  
            # 计算叶子结点的横坐标  
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW  
            # 绘制叶子结点  
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt, leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff), cntrPt, str(key))  
    # 计算纵坐标  
    plotTree.yOff = plotTree.yOff +1.0/plotTree.totalD 
    
ID3_main()

"""画出树形结构的决策树"""
def createPlot(inTree):
    # 定义一块画布(画布是自己的理解)
    fig = plt.figure(1,facecolor='white')
    # 清空画布
    fig.clf()
    # 定义横纵坐标轴，无内容
    axprops = dict(xticks=[],yticks=[])
    # 绘制图像，无边框，无坐标轴
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    # plotTree.totalW保存的是树的宽
    plotTree.totalW = float(getNumLeafs(inTree))
    # plotTree.totalD保存的是树的高
    plotTree.totalD = float(getTreeDepth(inTree))
    # 决策树起始横坐标
    plotTree.xOff = - 0.5 / plotTree.totalW #从0开始会偏右
#    print  plotTree.xOff
    # 决策树的起始纵坐标
    plotTree.yOff = 1.0
    # 绘制决策树
    plotTree(inTree,(0.5,1.0))
    # 显示图像
    plt.show()

# 定义决策树决策结果的属性，用字典来定义
# 下面的字典定义也可写作 decisionNode={boxstyle:'sawtooth',fc:'0.8'}
# boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细
decisionNode = dict(boxstyle="sawtooth",fc="0.8")
# 定义决策树的叶子结点的描述属性
leafNode = dict(boxstyle="round4",fc="0.8")
# 定义决策树的箭头属性
arrow_args = dict(arrowstyle="<-")

# 绘制结点
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
      #annotate是关于一个数据点的文本
      #nodeTxt为要显示的文本，centerPt为文本的中心点，箭头所在的点，parentPt为指向文本的点
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',
                            va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)

# 获得决策树的叶子结点数目
def getNumLeafs(myTree):
    # 定义叶子结点数目
    numLeafs = 0
    Nleft=0
    Nright=0
    if type(myTree['left2']).__name__ == 'dict':
        Nleft+=getNumLeafs(myTree['left2'])
    else:
        Nleft+=1
    if type(myTree['right3']).__name__ == 'dict':
        Nright += getNumLeafs(myTree['right3'])
    else:
         Nright += 1
        # 如果secondDict[key]为一个字典，即决策树结点
    numLeafs=Nleft+Nright
    return numLeafs

# 获得决策树的深度
def getTreeDepth(myTree):
    # 定义树的深度
    maxDepth = 0
    rightD=0
    leftD=0
    if type(myTree['left2']).__name__ == 'dict':
         leftD= 1 + getTreeDepth(myTree['left2'])
    else:
        leftD=1
    if type(myTree['right3']).__name__ == 'dict':
        rightD=1+getTreeDepth(myTree['right3'])
    else:
        rightD=1
    if leftD>rightD:
        maxDepth = leftD
    else:
        maxDepth=rightD
    # 返回树的深度
    return maxDepth

# 绘制决策树
def plotTree(myTree,parentPt):
    # 定义并获得决策树的叶子结点数
    numLeafs = getNumLeafs(myTree)
    #得到决策树的高度
    depth=getTreeDepth(myTree)
    # 得到第一个特征
    firstStr = myTree['spVal1']
    # 计算坐标，x坐标为当前树的叶子结点数目除以整个树的叶子结点数再除以2，y为起点
    cntrPt = (plotTree.xOff + (1.0 +float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    # 绘制中间结点，即决策树结点，也是当前树的根结点，这句话没感觉出有用来，注释掉照样建立决策树，理解浅陋了，理解错了这句话的意思，下面有说明
    # 绘制决策树结点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    # 根据firstStr找到对应的值
    if type(myTree['left2']).__name__ == 'dict':
    # 因为进入了下一层，所以y的坐标要变 ，图像坐标是从左上角为原点
        #plotTree.yOff = plotTree.xOff - 1.0/plotTree.totalD
        plotTree(myTree['left2'], cntrPt)
    else:
        plotTree.xOff = plotTree.xOff - 1.0 / plotTree.totalW
        plotNode(myTree['left2'], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
    if type(myTree['right3']).__name__ == 'dict':
    # 因为进入了下一层，所以y的坐标要变 ，图像坐标是从左上角为原点
        #plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
        plotTree(myTree['right3'], cntrPt)
    else:
        plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
        plotNode(myTree['right3'], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
    # 计算纵坐标
    plotTree.yOff = plotTree.yOff +1.0/plotTree.totalD



