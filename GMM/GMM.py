import LoadData.LoadWineData as DataLoader
import numpy as np
import math

data_x, data_y, _ = DataLoader.WineDataLoder()

print(data_y)


class GMM:
    def __init__(self, k, dim, data_x, data_y):
        self.k = k
        self.x = data_x
        self.y = data_y
        self.dim = dim
        self.a1 = 0.33
        self.a2 = 0.34
        self.a3 = 0.33
        self.u1 = np.sum(self.x[0:50], axis=0)/50
        self.u2 = np.sum(self.x[60:110], axis=0)/50
        self.u3 = np.sum(self.x[120:170], axis=0)/50
        self.segma1 = np.mat(self.x[0:50]-np.tile(self.u1, (50, 1))).T * np.mat(self.x[0:50]-np.tile(self.u1, (50, 1)))/50
        self.segma2 = np.mat(self.x[60:110]-np.tile(self.u2, (50, 1))).T * np.mat(self.x[60:110]-np.tile(self.u2, (50, 1)))/50
        self.segma3 = np.mat(self.x[120:170]-np.tile(self.u3, (50, 1))).T * np.mat(self.x[120:170]-np.tile(self.u3, (50, 1)))/50

    def gaussian(self, x):
        x = np.mat(x)
        u1 = np.mat(self.u1)
        u2 = np.mat(self.u2)
        u3 = np.mat(self.u3)
        g = abs(np.linalg.det(self.segma1))
        g1 = (1/math.sqrt(math.pow(2*math.pi, self.k)*np.abs(np.linalg.det(self.segma1))))
        g = np.array(-0.5*(x-u1)*self.segma1.I*(x-u1).T).squeeze()
        g1 = g1 *math.exp(-0.5*(x-u1)*self.segma1.I*(x-u1).T)
        g2 = (1/math.sqrt(math.pow(2*math.pi, self.k)*abs(np.linalg.det(self.segma2))))*math.exp(-0.5*(x-u2)*self.segma2.I*(x-u2).T)
        g3 = (1/math.sqrt(math.pow(2*math.pi, self.k)*abs(np.linalg.det(self.segma3))))*math.exp(-0.5*(x-u3)*self.segma3.I*(x-u3).T)
        return g1, g2, g3

    def onceRepeat(self):
        r = []
        for j in range(len(self.x)):
            g1, g2, g3 = self.gaussian(self.x[j])
            temp = []
            temp.append(self.a1 * g1 / (self.a1 * g1 + self.a2 * g2 + self.a3 * g3))
            temp.append(self.a2 * g2 / (self.a1 * g1 + self.a2 * g2 + self.a3 * g3))
            temp.append(self.a3 * g3 / (self.a1 * g1 + self.a2 * g2 + self.a3 * g3))
            r.append(temp)
        r = np.array(r)
        sum = np.sum(r, axis=0)
        self.u1 = np.sum([r[i][0]*np.array(self.x[i]) for i in range(len(self.x))], axis=0)/sum[0]
        self.segma1 = np.mat(np.array(self.x)-np.array(self.u1)).T * np.diag(np.array(r[:, 0:1]).squeeze()) * np.mat(np.array(self.x)-np.array(self.u1))/sum[0]
        self.a1 = sum[0]/len(self.x)

        self.u2 = np.sum([r[i][1]*np.array(self.x[i]) for i in range(len(self.x))], axis=0)/sum[1]
        self.segma2 = np.mat(np.array(self.x)-np.array(self.u2)).T * np.diag(np.array(r[:, 1:2]).squeeze()) * np.mat(np.array(self.x)-np.array(self.u2)) / sum[1]
        self.a2 = sum[1] / len(self.x)

        self.u3 = np.sum([r[i][2]*np.array(self.x[i]) for i in range(len(self.x))], axis=0)/sum[2]
        self.segma3 = np.mat(np.array(self.x)-np.array(self.u3)).T * np.diag(np.array(r[:, 2:3]).squeeze()) * np.mat(np.array(self.x)-np.array(self.u3)) / sum[2]
        self.a3 = sum[2] / len(self.x)

    def train(self, epoch):
        for e in range(epoch):
            self.onceRepeat()

    def predict(self, x):
        g1, g2, g3 = self.gaussian(x)
        if g1 > g2 and g1 > g3:
            return 1.0
        elif g2 > g1 and g2 > g3:
            return 2.0
        elif g3 > g1 and g3 > g2:
            return 3.0


model = GMM(3, 13, data_x, data_y)
model.train(100)
out = model.predict(data_x[160])
print(out)
count = 0
for i in range(len(data_x)):
    if model.predict(data_x[i]) != data_y[i]:
        count += 1
print(count)


