import numpy as np

data_x = []
data_y = []
with open("../DataSets/wine.data", 'r') as reader:
    items = reader.readlines()
    for item in items:
        data = list(map(float, item.split(',')))
        data_x.append(data[1:14])
        data_y.append(data[0])

print(data_y)


class LDA:
    def __init__(self, train_x, trian_y):
        self.w12 = []
        self.w13 = []
        self.w23 = []
        self.train_x1 = []
        self.train_x2 = []
        self.train_x3 = []
        self.train_y1 = []
        self.train_y2 = []
        self.train_y3 = []
        self.w12_l1 = 0
        self.w12_l2 = 0
        self.w13_l1 = 0
        self.w13_l3 = 0
        self.w23_l2 = 0
        self.w23_l3 = 0

        for i in range(len(trian_y)):
            if trian_y[i] == 1.0:
                self.train_x1.append(train_x[i])
                self.train_y1.append(1.0)
            if trian_y[i] == 2.0:
                self.train_x2.append(train_x[i])
                self.train_y2.append(2.0)
            if trian_y[i] == 3.0:
                self.train_x3.append(train_x[i])
                self.train_y3.append(3.0)

    def fit(self):
        u1 = np.array(self.train_x1).sum(axis=0)/len(self.train_x1)
        u2 = np.array(self.train_x2).sum(axis=0)/len(self.train_x2)
        u3 = np.array(self.train_x3).sum(axis=0)/len(self.train_x3)
        sum1 = np.mat(np.array(self.train_x1)-np.array(u1)).T * np.mat(np.array(self.train_x1)-np.array(u1))
        sum2 = np.mat(np.array(self.train_x2)-np.array(u2)).T * np.mat(np.array(self.train_x2)-np.array(u2))
        sum3 = np.mat(np.array(self.train_x3)-np.array(u3)).T * np.mat(np.array(self.train_x3)-np.array(u3))
        self.w12 = (sum1 + sum2).I * np.mat(u1 - u2).T
        self.w13 = (sum1 + sum3).I * np.mat(u1 - u3).T
        self.w23 = (sum2 + sum3).I * np.mat(u2 - u3).T
        self.w12_l1 = (self.w12.T * np.mat(u1).T)[0, 0]
        self.w12_l2 = (self.w12.T * np.mat(u2).T)[0, 0]
        self.w13_l1 = (self.w13.T * np.mat(u1).T)[0, 0]
        self.w13_l3 = (self.w13.T * np.mat(u3).T)[0, 0]
        self.w23_l2 = (self.w23.T * np.mat(u2).T)[0, 0]
        self.w23_l3 = (self.w23.T * np.mat(u3).T)[0, 0]
        print(self.w12.T)
        print(self.w13.T)
        print(self.w23.T)

    def predict(self, train_x):
        out = []
        for x in train_x:
            len1 = (self.w12.T * np.mat(x).T)[0, 0]
            if abs(len1-self.w12_l1) < abs(len1-self.w12_l2):
                len3 = (self.w13.T * np.mat(x).T)[0, 0]
                if abs(len3 - self.w13_l1) < abs(len3 - self.w13_l3):
                    out.append(1.0)
                else:
                    out.append(3.0)
            else:
                len2 = (self.w23.T * np.mat(x).T)[0, 0]
                if abs(len2 - self.w23_l2) < abs(len2 - self.w23_l3):
                    out.append(2.0)
                else:
                    out.append(3.0)
        return out

    def loss(self, x, y):
        count = 0
        yy = self.predict(x)
        for i in range(len(y)):
            if yy[i] != y[i]:
                count += 1
        return count


model = LDA(data_x, data_y)
model.fit()
print(model.loss(data_x, data_y))
