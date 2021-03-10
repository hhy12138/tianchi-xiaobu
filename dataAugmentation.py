import math
from utils.read import read
from config.config import train_dir
def dataAug(X,Y):
    graph = dict()
    for x,y in zip(X,Y):
        left = x[0]
        right = x[1]
        left_str = ','.join([str(w) for w in left])
        right_str = ','.join([str(w) for w in right])
        if left_str not in graph:
            graph[left_str] = {0:set(),1:set()}
        if right_str not in graph:
            graph[right_str] = {0:set(),1:set()}
        graph[left_str][y].add(right_str)
        graph[right_str][y].add(left_str)
    new_X = []
    new_Y = []
    completed = set()
    for item in graph.keys():
        if item not in completed:
            ones = set()
            zeros = set()
            oneStack = []
            oneStack.append(item)
            while len(oneStack) > 0:
                origin = oneStack[-1]
                for target in graph[item][0]:
                    zeros.add(target)
                oneStack.pop(-1)
                ones.add(origin)
                for target in graph[item][1]:
                    if target not in ones:
                        oneStack.append(target)
            ones_list = list(ones)
            for id,left in enumerate(ones_list[:-1]):
                for right in ones_list[id+1:]:
                    leftn = list(map(lambda x:int(x),left.split(',')))
                    rightn = list(map(lambda x:int(x),right.split(',')))
                    new_X.append([leftn,rightn])
                    new_Y.append(1)
            # for left in ones_list:
            #     for right in zeros:
            #         leftn = list(map(lambda x:int(x),left.split(',')))
            #         rightn = list(map(lambda x: int(x), right.split(',')))
            #         new_X.append([leftn, rightn])
            #         new_Y.append(0)
            for c in ones:
                completed.add(c)
    for x, y in zip(X, Y):
        if y == 0:
            new_X.append(x)
            new_Y.append(y)
    return new_X,new_Y
if __name__=='__main__':
    X,Y = read(train_dir,mode='train')
    a = dataAug(X,Y)
    print(1)
