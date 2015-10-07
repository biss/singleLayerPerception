__author__ = 'biswajeet'

import re

class Em:
    w = ''
    def __init__(self):
        self.w = 'hello'
        self.var = raw_input("Please enter the sl no.")

    def t_print(self):
        one = re.split('\s|[,.]', self.var)
        print 'your entered sl no is ',one[0], one[1]

m = Em()
m.t_print()