import numpy


def WineDataLoder():
    data_x = []
    data_y = []
    data_xy = []
    with open("../DataSets/wine.data", 'r') as reader:
        items = reader.readlines()
        for item in items:
            data = list(map(float, item.split(',')))
            data_x.append(data[1:])
            data_y.append(data[0])
            data_xy.append(data[1:] + data[0:1])
    return data_x, data_y, data_xy