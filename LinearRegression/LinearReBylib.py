import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib as mpl

# 加载数据
data_x = []
data_y = []
with open("../DataSets/housing.data", 'r') as reader:
    items = reader.readlines()
    for item in items:
        data = list(map(float, item.split()))
        data_x.append(data[0:13])
        data_y.append(data[13])
print(data_x[:1])

# 选择线性回归模型
model = LinearRegression()
# 训练模型
model.fit(data_x[50:], data_y[50:])

# 打印模型参数
print("----------------模型参数-------------------")
print(model.coef_)  # 这里打印权值
print(model.intercept_)  # 这里打印偏移量
print("----------------模型参数-------------------")
y_p = model.predict(data_x)
loss = np.array((np.mat(y_p)-np.mat(data_y))*(np.mat(y_p)-np.mat(data_y)).T).squeeze() /(2*len(y_p))
temp_x = np.array(data_x[:50])   # 这里我们用前50个数据来进行预测


# 这里我们采用房子的第一个属性 来显示与房价的关系
# plt.scatter(temp_x[:, :1], model.predict(temp_x), color='r', label='predicition')  # 这里绘制预测的房价
# plt.scatter(temp_x[:, :1], data_y[:50], color='b', label='original_data')  # 这里预测实际的房价
# print(loss)  # 这里给我们的训练打分
# plt.show()

#设置matplotlib 支持中文显示
mpl.rcParams['font.family'] = 'SimHei' #设置字体为黑体
mpl.rcParams['axes.unicode_minus'] = False #设置在中文字体是能够正常显示负号（“-”）

plt.figure(figsize=(10,10))
#绘制预测值
plt.plot(model.predict(data_x[:100]),'ro-',label="预测值")
plt.plot(data_y[:100],'go--',label="真实值")
plt.title("线性回归预测-梯度下降法")
plt.xlabel("样本序号")
plt.ylabel("房价")
plt.legend()
plt.show()


# [2.56871037e-05 1.68607800e-04 1.11564080e-04 9.95400000e-07
#  6.09442791e-06 7.39240776e-05 7.32581140e-04 4.57138742e-05
#  9.34451000e-05 4.28779760e-03 2.05316730e-04 4.20804951e-03
#  1.19799159e-04]
