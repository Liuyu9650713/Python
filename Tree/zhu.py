# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:25:46 2019

@author: Administrator
"""
import trees
from imp import reload 
reload(trees)
myDat,labels=trees.createDataSet()
myTree=trees.createTree(myDat,labels)
print(myTree)
#myTree[]['no surfacing'][3]='maybe'
#mytree{}