import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
Batch = 10
data_x = []
data_y = []
with open("../DataSets/housing.data", 'r') as reader:
    items = reader.readlines()
    for item in items:
        data = list(map(float, item.split()))
        data_x.append(data[0:13])
        data_y.append(data[13])
data_xn = np.sum(data_x, axis=0)/len(data_x)
data_xmax = np.argmax(data_x, axis=0)
data_xmin = np.argmin(data_x, axis=0)
data_yn = np.sum(data_y)/len(data_y)
data_ymax = np.max(data_y)
data_ymin = np.min(data_y)
print(data_x[0])
data_xx = (data_x - data_xn)/(data_xmax-data_xmin)
data_yy = (data_y - data_yn)/(data_ymax-data_ymin)


class DataType:
    def __init__(self):
        self.data_x = []
        self.data_y = []


class DataGenerator:
    def __init__(self, data_xx, data_yy, batch_size):
        self.x = data_xx
        self.y = data_yy
        self.batch_size = batch_size
        self.batch_data = []

        self.data_shuffle()
        self.data_prepare()

    def data_shuffle(self):
        index = [i for i in range(len(self.x))]
        np.random.shuffle(index)  # 对索引进行打乱
        x = np.array(self.x)[index]
        y = np.array(self.y)[index]
        self.x = x
        self.y = y

    def data_prepare(self):
        index = 0
        while True:
            temp = DataType()
            if index + self.batch_size >= len(self.x):
                temp.data_x = self.x[index:]
                temp.data_y = self.y[index:]
                self.batch_data.append(temp)
                break
            else:
                temp.data_x = self.x[index:index + self.batch_size]
                temp.data_y = self.y[index:index + self.batch_size]
                index += self.batch_size
                self.batch_data.append(temp)

    def get_batch(self):
        while True:
            for each in self.batch_data:
                yield each


class LinearRegression:
    def __init__(self, dim, lr):
        self.weights = np.zeros(dim)
        self.lr = lr
        self.bias = np.random.randn(1, 1)
        self.thet = []
        self.p1 = 0.9
        self.p2 = 0.9999
        self.s = np.zeros(dim+1)
        self.t = 0
        self.r = np.zeros(dim+1)

    def get_weight(self):
        return self.weights

    def get_bia(self):
        return self.bias

    def fit_lss(self, train_data_x, train_data_y):
        b = np.ones(len(train_data_x))
        x = np.mat(train_data_x)
        x = np.c_[x, b]
        y = np.mat(train_data_y)
        out = (x.T * x).I * x.T * y.T
        self.thet = np.array(out).squeeze()
        return np.array(out).squeeze()

    def predict_lss(self, train_data_x):
        b = np.ones(len(train_data_x))
        x = np.mat(train_data_x)
        x = np.c_[x, b]
        out = self.thet.T.dot(x.T)
        return np.array(out).squeeze()

    def loss_lss(self, train_data_x, train_data_y):
        y = np.mat(train_data_y)
        y_p = np.mat(self.predict_lss(train_data_x))
        out = np.array((y_p-y) * (y_p-y).T).squeeze()
        return out/(2*len(train_data_x))

    def get_grad(self, train_data_x, train_data_y):
        my_theta = np.mat(np.append(self.weights, self.bias))
        b = np.ones(len(train_data_x))
        x = np.mat(train_data_x)
        x = np.c_[x, b]
        y = np.mat(train_data_y)
        grad = np.array(x.T*(x*my_theta.T-y.T)*self.lr).squeeze()
        return grad

    def update_grad(self, grad):
        self.weights = self.weights - grad[0:13]
        self.bias = self.bias - grad[-1]

    def predict_gd(self, train_data_x):
        my_theta = np.mat(np.append(self.weights, self.bias))
        b = np.ones(len(train_data_x))
        x = np.mat(train_data_x)
        x = np.c_[x, b]
        out = my_theta * x.T
        return np.array(out).squeeze()

    def loss_gd(self, train_data_x, train_data_y):
        y = np.mat(train_data_y)
        y_p = np.mat(self.predict_gd(train_data_x))
        out = np.array((y_p - y) * (y_p - y).T).squeeze()
        return out / (2 * len(train_data_x))


# 准备模型
model = LinearRegression(13, lr=0.00000001)
# 使用最小二乘法拟合
output = model.fit_lss(data_x, data_y)
print(output)
output = model.predict_lss(data_x[0:10])
print(output)
print(model.loss_lss(data_x, data_y))

# 准备批数据生产器
G = DataGenerator(data_x, data_y, batch_size=32)
train_data = G.get_batch()
# 使用梯度下降训练模型
for i in range(500000):
    train = next(train_data)
    x_train = train.data_x
    y_train = train.data_y
    grad = model.get_grad(x_train, y_train)
    model.update_grad(grad)
    if i % 10000 == 0:
        print(model.loss_gd(data_x, data_y))


# #绘制预测值
mpl.rcParams['font.family'] = 'SimHei'  # 设置字体为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 设置在中文字体是能够正常显示负号（“-”）
plt.figure(figsize=(10, 10))
plt.plot(model.predict_gd(data_x[:100]),'ro-',label="预测值")
plt.plot(data_y[:100], 'go--', label="真实值")
plt.title("线性回归预测-梯度下降法")
plt.xlabel("样本序号")
plt.ylabel("房价")
plt.legend()
plt.show()

