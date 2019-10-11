import LoadData.LoadWineData as DataLoader
import numpy as np
import math

data_x, data_y, _ = DataLoader.WineDataLoder()

print(data_y)


class KMeans:
    def __init__(self, k, data_x, data_y, dim):
        self.k = k
        self.x = data_x
        self.y = data_y
        self.dim = dim
        self.u1 = data_x[0]
        self.u2 = data_x[int(len(data_x)/2)]
        self.u3 = data_x[len(data_x)-1]

    def getDistance(self, x1, x2):
        dist = 0
        for i in range(len(x1)):
            # dist += (abs(x1[i] - x2[i]) * abs(x1[i] - x2[i]))
            dist += (abs(x1[i] - x2[i]) / (x1[i] + x2[i])) * (abs(x1[i] - x2[i]) / (x1[i] + x2[i])) * 4
        return math.sqrt(dist)

    def onceRepeat(self):
        u1_class = []
        u2_class = []
        u3_class = []
        for each in self.x:
            dist1 = self.getDistance(each, self.u1)
            dist2 = self.getDistance(each, self.u2)
            dist3 = self.getDistance(each, self.u3)
            if dist1 < dist2 and dist1 < dist3:
                u1_class.append(each)
            elif dist2 < dist1 and dist2 < dist3:
                u2_class.append(each)
            elif dist3 < dist1 and dist3 < dist2:
                u3_class.append(each)
        self.u1 = np.sum(u1_class, axis=0)/len(u1_class)
        self.u2 = np.sum(u2_class, axis=0)/len(u2_class)
        self.u3 = np.sum(u3_class, axis=0)/len(u3_class)

    def train(self):
        while True:
            temp1 = self.u1
            temp2 = self.u2
            temp3 = self.u3
            self.onceRepeat()
            if temp1[0] == self.u1[0] and temp2[0] == self.u2[0] and temp3[0] == self.u3[0]:
                break

    def predict(self):
        out = []
        for each in self.x:
            dist1 = self.getDistance(each, self.u1)
            dist2 = self.getDistance(each, self.u2)
            dist3 = self.getDistance(each, self.u3)
            if dist1 < dist2 and dist1 < dist3:
                out.append(1.0)
            elif dist2 < dist1 and dist2 < dist3:
                out.append(2.0)
            elif dist3 < dist1 and dist3 < dist2:
                out.append(3.0)
        return out


model = KMeans(3, data_x, data_y, 13)
model.train()
out = model.predict()
count = 0
for i in range(len(data_x)):
    if out[i] != data_y[i]:
        count += 1
print(count)
