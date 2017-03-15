'''
Created on Mar 15, 2017
kNN: k Nearest Neighbors

'''
from numpy import *


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=lambda asd:asd[1], reverse=True)
    return sortedClassCount[0][0]

if __name__=="__main__":
    dataSet=[[1,1],[1,1.1],[0,0],[0,0.1]]
    labels=['A','A','B','B']
    index=[0,0.1]
    print classify0(index,dataSet,labels,3)
    

