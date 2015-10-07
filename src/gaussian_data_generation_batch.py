__author__ = 'biswajeet'

""" class 1 vs class 2 distinguished by mean1 and mean2 respectively
    class1 < 0 and class2 >= 0  """

import numpy as np
import matplotlib.pyplot as plt

def show_plot(formula, x_range):
    plt.plot(x_data,y_data, 'x')
    y = eval(formula)
    plt.plot(x, y, color = 'Red')
    plt.axis('equal')
    plt.show()

mean1 = [0.0, 10.0]
mean2 = [20.0, 15.0]

cov1 = [[8.0, 3.0],
       [3.0, 8.0]] # diagonal covariance, points lie on x or y-axis

cov2 = [[5.0, -10.0],
       [-10.0, 15.0]]

x,y = np.random.multivariate_normal(mean1,cov1,100).T

x1,y1 = np.random.multivariate_normal(mean2,cov2,100).T

x_data = np.concatenate((x, x1), axis = 0)
y_data = np.concatenate((y, y1), axis = 0)

label1 = np.ones(100)
label2 = label1 + 1

labels = np.concatenate((label1, label2), axis = 0)

"""data_x = x.tolist()
for ele in x1:
    data_x.append(ele)

data_y = y.tolist()
for ele in y1:
    data_y.append(ele)

plt.plot(x_data,y_data, 'x')
plt.axis('equal')"""

#print(len(x_data))
x_range = range(1, 8)

u_weights = np.array([-1.0, 5.0, 3.0])        #bias, x-coefficient, y-coefficient
l_weights = np.array([0.0, 0.0, 0.0])       #numpy array([ 0.,  0.,  0.])

final_data = np.zeros(shape = (200, 4)) #numpy array of 200 rows and 4 columns

#putting data in numpy array
for i, x, y, l in zip(range(200), x_data, y_data, labels):
    final_data[i] = [1, x, y, l]

#print final_data
#print np.transpose(u_weights)
pdt = 0
eta = 0.1
x_val = np.linspace(1,8,400)
batch_size = 0
batch_del_w = np.zeros(shape=(20, 3))

#class1 < 0 and class2 >= 0 - online learning
while(abs(l_weights - u_weights).any() > .0001):
    l_weights = u_weights

    for i in final_data:
        #print (u_weights, i[:-1])
        batch_size += 1
        pdt = np.dot(i[:-1],np.transpose(u_weights))
        #print pdt
        if i[-1] == 1 and pdt >= 0:
            print 'storing '
            print(- eta*i[:-1])
            print batch_size
            batch_del_w[batch_size - 1] = (- eta*i[:-1])
            #u_weights = u_weights - eta*i[:-1]
            #print (u_weights, i[:-1])

        if i[-1] == 2 and pdt < 0:
            batch_del_w[batch_size - 1] = ( eta*i[:-1])
            #u_weights = u_weights + eta*i[:-1]
            #print (u_weights, i[:-1])
        if batch_size == 20:
            mul = eta*(np.sum(batch_del_w, axis = 0))/20
            print 'batch del and mul', batch_del_w, mul
            u_weights = u_weights + mul
            print 'new u_weights', u_weights
            batch_size = 0
    print 'weight diff',abs(l_weights - u_weights)
formula = '(-u_weights[1]/u_weights[2])*x_val - u_weights[0]/u_weights[2]'
show_plot(formula, x_range)

"""
74
70
79
70
"""



