from numpy import *
import regTrees as re

def Model_Tree():
    data = re.loadDataSet('exp2.txt')
    myMat = mat(data)
    tree = re.createTree(myMat,modelLeaf,modelErr)
    print(tree)
#作用：将数据集格式化成目标变量Y和自变量X，并求回归系数向量
# 输入：数据集
# 输出：回归系数向量，自变量，目标变量
def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n - 1]  # X[:, 0]为截距，均为1
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
                        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


# 作用：求数据集的回归系数向量
# 输入：数据集
# 输出：回归系数向量
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws


# 作用：求数据集的方差
# 输入：数据集
# 输出：数据集的方差
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


#Model_Tree()