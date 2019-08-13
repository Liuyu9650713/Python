# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 17:30:53 2019

@author: Administrator
"""

def getentropy(tree,dataSet,lables):
    countTree={}
    firstStr = list(tree.keys())[0]#获取第一个键值
    p=labels.index(firstStr)
    countTree[firStr]=