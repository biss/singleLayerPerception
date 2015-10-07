__author__ = 'biswajeet'
__Date__   = '30/08/2015'

""" class 1 vs class 2 distinguished by mean1 and mean2 respectively
    class1 < 0 and class2 >= 0  """

import re
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as cma

class CL_Perceptron_2Class_2D(object):

    def __init__(self):
        self.mean_1 = raw_input('Specify mean of the first gaussian(in 2D):')
        self.mean_2 = raw_input('Specify mean of the second gaussian(in 2D):')

        self.var_cov_1 = raw_input('Specify variance-covariance matrix of the first gaussian(in 2D):')
        self.var_cov_2 = raw_input('Specify variance-covariance matrix of the second gaussian(in 2D):')

    def construct_mean_var(self):

        temp_mean = [int(x) for x in re.split('\s|[,.]', self.mean_1)]
        self.mean_1 = temp_mean
        temp_mean = [int(x) for x in re.split('\s|[,.]', self.mean_2)]
        self.mean_2 = temp_mean

        temp_cv = [int(x) for x in re.split('\s|[,.]', self.var_cov_1)]
        self.var_cov_1 = [[temp_cv[0], temp_cv[1]],[temp_cv[2], temp_cv[3]]]
        temp_cv = [int(x) for x in re.split('\s|[,.]', self.var_cov_2)]
        self.var_cov_2 = [[temp_cv[0], temp_cv[1]],[temp_cv[2], temp_cv[3]]]

    def print_CV_matrix(self):
        print 'mean of the two gaussians', self.mean_1, self.mean_2
        print 'var-cov of the 2 gaussians', self.var_cov_1, self.var_cov_2

    def construct_dataset(self):


a = CL_Perceptron_2Class_2D()
a.construct_mean_var()
a.print_CV_matrix()


"""mean1 = [5,20]
mean2 = [25, 10]

cov1 = [[10, 0],
       [0, 10]] # diagonal covariance, points lie on x or y-axis

cov2 = [[5,-10],
       [-10,15]]

x_0,y_0 = np.random.multivariate_normal(mean1,cov1,100).T
x1,y1 = np.random.multivariate_normal(mean2,cov2,100).T

x_data = np.concatenate((x_0, x1), axis = 0)
y_data = np.concatenate((y_0, y1), axis = 0)

label1 = np.ones(100)
label2 = label1 + 1

labels = np.concatenate((label1, label2), axis = 0)

def show_plot(formula, x_range):
    plt.plot(x_0, y_0, 'x', color = 'Green')
    plt.plot(x1, y1, 'x', color = 'Blue')
    y_range = eval(formula)
    plt.plot(x_range, y_range, color = 'Red')
    plt.axis('equal')
    plt.show()

#print(len(x_data))

u_weights = np.array([-1, 5, 3])
l_weights = np.zeros(shape = (3))

final_data = np.zeros(shape = (200, 4))

plt.plot(x_0, y_0, 'x', color = 'Green')
plt.plot(x1, y1, 'x', color = 'Blue')
plt.show()

#preparing training and testing data sets
for i, x, y, l in zip(range(200), x_data, y_data, labels):
    final_data[i] = [1, x, y, l]

#print final_data
pdt = 0
eta = .1
x_range = np.linspace(-15, 30, 400)
formula = '(-u_weights[1]/u_weights[2])*x_range - u_weights[0]/u_weights[2]'

#class1 < 0 and class2 >= 0 - online learning
while(abs(l_weights - u_weights).any() > .0001):
    l_weights = u_weights

    #show_plot(formula, x_range)

    for i in final_data:
        print (u_weights, i[:-1])
        pdt = np.dot(i[:-1],np.transpose(u_weights))
        print pdt
        if i[-1] == 1 and pdt >= 0:
            u_weights = u_weights - eta*i[:-1]
            print (u_weights, i[:-1])

        if i[-1] == 2 and pdt < 0:
            u_weights = u_weights + eta*i[:-1]
            print (u_weights, i[:-1])
    print abs(l_weights - u_weights)
    show_plot(formula, x_range)
show_plot(formula, x_range)

#generating test data

x_test,y_test = np.random.multivariate_normal(mean1,cov1,100).T

x1_test,y1_test = np.random.multivariate_normal(mean2,cov2,100).T

x_data_test = np.concatenate((x_test, x1_test), axis = 0)
y_data_test = np.concatenate((y_test, y1_test), axis = 0)

label1_test = np.ones(100)
label2_test = label1_test + 1

labels_test = np.concatenate((label1_test, label2_test), axis = 0)
#print label1_test.shape

test_data = np.zeros(shape = (200, 3))

#preparing training and testing data sets
for i, x, y in zip(range(200), x_data_test, y_data_test):
    test_data[i] = [1, x, y]

prediction = np.zeros(shape=(200))

t_pdt = 0
count = 0
for i in test_data:
    t_pdt = np.dot(i[:],np.transpose(u_weights))
    if t_pdt >= 0:
        prediction[count] = 2
    else:
        prediction[count] = 1
    count += 1
print prediction
cm = cma.confusion_matrix(labels_test, prediction)
print 'confusion matrix for the model is', cm

87
74
70
79
70

"""