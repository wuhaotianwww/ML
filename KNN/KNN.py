import LoadData.LoadWineData as DataLoader
import numpy as np
import queue
import math

data_x, data_y, data_xy = DataLoader.WineDataLoder()
# 数据预处理
data_x_span = np.max(data_x, axis=0) - np.min(data_x, axis=0)
data_x = ((np.array(data_x) - np.sum(data_x, axis=0)/len(data_x))/data_x_span*10.0).tolist()

data_xy = np.hstack((data_x, np.array(data_y).reshape(178, -1))).tolist()
print(data_y)


class TreeNode(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.father = None
        self.feature = []
        self.label = 0


class KNN:
    def __init__(self, dim, k):
        self.dim = dim
        self.k = k
        self.root = None

    def CreateKDTree(self, train, deep):
        if len(train) == 0:
            return None
        root = TreeNode()
        deep = deep % self.dim
        train.sort(key=lambda x: x[deep])
        index = int(len(train)/2)
        while index+1 < len(train) and train[index][deep] == train[index+1][deep]:
            index += 1
        if len(train) != 0:
            root.label = train[index][13]
            root.feature = train[index][0:13]
            root.left = self.CreateKDTree(train[0:index], deep + 1)
            if root.left:
                root.left.father = root
            root.right = self.CreateKDTree(train[index+1:], deep + 1)
            if root.right:
                root.right.father = root
        self.root = root
        return root

    def get_dist(self, x1, x2):
        dist = 0
        for i in range(self.dim):
            dist += (abs(x1[i] - x2[i])*abs(x1[i]-x2[i]))
        return math.sqrt(dist)

    def isinsect(self, node, x, cmp):
        y = node.feature
        for i in range(self.dim):
            if abs(y[i] - x[i]) < cmp:
                return True
        return False

    def dfs(self, node, nodelist):
        if node:
            nodelist.append(node)
        else:
            return nodelist
        if node.left:
            self.dfs(node.left, nodelist)
        if node.right:
            self.dfs(node.right, nodelist)
        return nodelist

    def SearchKDTree(self, x):
        deep = 0
        isfromleft = True
        tempnode = self.root
        while tempnode.right or tempnode.left:
            if x[deep] > tempnode.feature[deep] and tempnode.right:
                tempnode = tempnode.right
            elif tempnode.left:
                tempnode = tempnode.left
            else:
                tempnode = tempnode.right
            if tempnode == None:
                break
            deep = (deep+1) % self.dim

        q = queue.PriorityQueue(maxsize=self.k)
        dist = -1 * self.get_dist(tempnode.feature, x)
        q.put((dist, tempnode))
        if tempnode.father.left == tempnode:
            isfromleft = True
        else:
            isfromleft = False
        tempnode = tempnode.father

        while tempnode.father:
            temp = q.get()
            cmp = -1.0*temp[0]
            q.put(temp)
            if self.isinsect(tempnode, x, cmp) or q.qsize() < self.k:
                nodelist = [tempnode]
                if isfromleft:
                    self.dfs(tempnode.right, nodelist)
                else:
                    self.dfs(tempnode.left, nodelist)
                for each in nodelist:
                    dist = -1 * self.get_dist(each.feature, x)
                    if q.qsize() == self.k:
                        temp = q.get()
                        tempdist = temp[0]
                        if dist < tempdist:
                            q.put(temp)
                        else:
                            q.put((dist, tempnode))
                    else:
                        q.put((dist, tempnode))

            if tempnode.father.left == tempnode:
                isfromleft = True
            else:
                isfromleft = False
            tempnode = tempnode.father

        out = []
        while q.qsize():
            out.append(q.get()[1].label)
        return out

    def decision(self, vote):
        dict = {1.0: 0, 2.0: 0, 3.0: 0}
        for key in vote:
            dict[key] = dict.get(key, 0) + 1
        if dict[1.0] >= dict[2.0] and dict[1.0] >= dict[3.0]:
            label = 1.0
        elif dict[2.0] >= dict[1.0] and dict[2.0] >= dict[3.0]:
            label = 2.0
        else:
            label = 3.0
        return label


model = KNN(13, 3)
root = model.CreateKDTree(data_xy, 0)
count = 0
result = []
for i in range(len(data_x)):
    out = model.SearchKDTree(data_x[i])
    out = model.decision(out)
    result.append(out)
    if out != data_y[i]:
        count += 1
print(result)
print(count)
