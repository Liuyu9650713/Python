from numpy import *
import regTrees as re
import Tree_Model as tr

def SL_Model():
    test=re.loadDataSet('bikeSpeedVsIq_test.txt')
    train=re.loadDataSet('bikeSpeedVsIq_train.txt')
    testMat=mat(test)
    trainMat=mat(train)

    mytree=re.createTree(trainMat,ops=(1,20))
    yHat = createForeCast(mytree, testMat[:, 0])
    print('回归树的相关性R^2：',corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])

    myTree=re.createTree(trainMat,tr.modelLeaf,tr.modelErr,ops=(1,20))
    yHat = createForeCast(myTree, testMat[:, 0],modelTreeEval)
    print('模型树的相关性R^2：',corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])

    ws, X, Y = tr.linearSolve(trainMat)
    for i in range(shape(testMat)[0]):
        yHat[i]=testMat[i,0]*ws[1,0]+ws[0,0]
    print('标准线性回归的相关性R^2：',corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])



# 作用：对回归树叶节点进行预测
# 输入：树模型
# 输出：模型的浮点数
def regTreeEval(model, inDat):
    return float(model)

# 作用：对模型树叶节点进行预测
# 输入：树模型，测试数据
# 输出：预测值
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


# 作用：预测测试数据在树中的值，自顶向下遍历整个树，直到命中叶节点为止，输入单个数据点/行向量。
# 输入：数，测试数据，树形式
# 输出：测试数据在树中的值
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not re.isTree(tree):
        return modelEval(tree, inData)
    # inData[]值比根节点大，进入左子树
    if inData[tree['spInd0']] > tree['spVal1']:
        if re.isTree(tree['left2']):
            return treeForeCast(tree['left2'], inData, modelEval)
        else:
            return modelEval(tree['left2'], inData)
    else:  # inData[]值比根节点小，进入右子树
        if re.isTree(tree['right3']):
            return treeForeCast(tree['right3'], inData, modelEval)
        else:
            return modelEval(tree['right3'], inData)


# 作用：#对数据进行树结构建模
# 输入：树，测试数据集，树形式
# 输出：模型树
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

SL_Model()