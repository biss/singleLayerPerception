__author__ = 'biswajeet'
"""import sys

def func(a = None):
    if a == None:
        print 'no string input'

    for i in range(len(a)):
        if(a[i] != '-'):
            sys.stdout.write(a[i])
        else:
            print_in_between(a[i-1], a[i+1])
            i += 1

def print_in_between(x, y):
    len = ord(y) - ord(x)
    if(len > 0):
        for i in range(ord(x)+1, ord(y), 1):
            sys.stdout.write(chr(i))
    

ip = raw_input()
func(ip)"""
import sys
main_arr = 'dnabcdedcba'
diff_list = [2, 3, 1, 1, 1, 1, -1, -1, -1, -1]
i = 0
j = 0
while(i < len(diff_list)):
    #increasing case
    #print 'iter ', i
    start_index = 0
    end_index = 0

    if (diff_list[i] == 1):
        start_index = i
        end_index = i
        while end_index < len(diff_list) and diff_list[end_index] == 1:
            end_index += 1
        i = end_index
        #print 'increasing len is ' ,(end_index - 1, start_index)

    length = end_index - start_index
    c_end = end_index
    c_start = start_index
    #print 'length is', length
    if(length > 0):
        #print start_index, end_index
        sys.stdout.write(main_arr[start_index]+'-'+main_arr[end_index])
    length = 0
    #decreasing
    if (diff_list[i] == -1):
        start_index = i
        end_index = i
        while end_index < len(diff_list) and diff_list[end_index] == -1:
            end_index += 1
        i = end_index
        #print 'decreasing len is ' ,(end_index - 1, start_index)
    length = end_index - start_index
    #print 'length is', length
    if(length > 0) and c_start != start_index and c_end != end_index:
        #print start_index, end_index
        sys.stdout.write(main_arr[start_index]+'-'+main_arr[end_index])
    else:
        sys.stdout.write(main_arr[i])
    i += 1


