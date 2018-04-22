#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-12 10:30:54
# @Author  : mianhk (yugc666@163.com)
# @Link    : ${link}
# @Version : $Id$


import math
import random

OPT_OBJECT = {
    'CPU': 0,
    'MEM': 1
}
# 预测时间粒度
# 训练数据粒度
TIME_GRAIN_HOUR = 0
TIME_GRAIN_DAY = 1


# 训练时间粒度
TRAIN_GRAIN_DAYS=7

# 检查dict中是否存在key
def isContainKey(dic, key):
    return key in dic.keys()


def calcu_days(begin_day,end_day):
    predict_time_grain=0
    st_year, st_month, st_day = begin_day.split(' ')[0].split('-')
    et_year, et_month, et_day = end_day.split(' ')[0].split('-')
    day_index = (int(et_year) - int(st_year)) * 365 + \
                                 (int(et_month) - int(st_month)) * 30 + (int(et_day) - int(st_day))
    return day_index
    # pass

def get_grain_time(time_str, time_grain):
    # 获取粒度时间
    split_append_tmp = [[13, ':00:00'], [10, ' 00:00:00']]
    sp_len_tmp = split_append_tmp[time_grain][0]
    sp_str_tmp = split_append_tmp[time_grain][1]
    return time_str[:sp_len_tmp] + sp_str_tmp;


def get_train_data1(case, flavor_num,move_length, input_num):
    # print 'flavor_num: ', flavor_num
    # print 'case.his_data_all[flavor_num]: ', case.his_data_all[flavor_num]
    his_array_size = case.train_data_range_days - case.predict_data_range_days + 1
    # move_length=int(case.predict_data_range_days/3)
    # move_length = 1
    # print 'move_length: ',move_length
    his_array_size = int(math.ceil((his_array_size + 0.0) / move_length))
    # predict_size=4
    his_array = [0 for x in range(his_array_size)]
    print 'his_array size: ',len(his_array)
    print 'his_array: ',his_array
    #删除大的离谱的数
    sum=0.0
    max=0.0
    for x in range(len(case.his_data_all[flavor_num])):
        if(max<case.his_data_all[flavor_num][x]):
            max=case.his_data_all[flavor_num][x]
            max_number=x
        sum+=case.his_data_all[flavor_num][x]
    average=sum/len(case.his_data_all)
    case.his_data_all[flavor_num][max_number]=2*average
    # print 'average: ',average
    move_ptr = 0
    while move_ptr <= (case.train_data_range_days - case.predict_data_range_days):
        temp = 0
        for j in range(case.predict_data_range_days):
            tt=case.his_data_all[flavor_num][move_ptr + j]
            if(tt>3*average and tt>2):
                tt-=3*average
            temp += tt
        # print 'move_ptr: ',move_ptr,'  temp: ',temp
        his_array[move_ptr / move_length] = temp
        move_ptr += move_length
    print "his_array处理", his_array
    #归一化处理
    mimi, maxi = getm(his_array)
    mimi = float(mimi)
    maxi = float(maxi)
    # print "max   ", maxi, "  mimi  ", mimi
    for flavor_num in range(len(his_array)):
        if(maxi-mimi<=0.1):
            his_array[flavor_num]=0.0
        else:
            his_array[flavor_num] = float((his_array[flavor_num] - mimi) / (maxi - mimi))
    # print "his_array", his_array
    # his_array = test_lof.ErrorDection(his_array, ave, de)
    train_data = [0 for x in range(len(his_array) - input_num)]
    # print 'his_array: '
    # print his_array
    for p in range(len(his_array) - input_num):
        # print 'p'
        # print p
        train_data[p] = his_array[p:p + input_num]
    print 'train_data:',train_data
    test_data = [[0] for x in range(input_num, len(his_array))]
    for x in range(len(his_array) - input_num):
        # print 'x'
        # print x
        test_data[x][0] = his_array[x + input_num]
    print 'test_data:', test_data
    return train_data, test_data, mimi, maxi,his_array

def get_train_data2(case, flavor_num,move_length, input_num):
    # print 'flavor_num: ', flavor_num
    # print 'case.his_data_all[flavor_num]: ', case.his_data_all[flavor_num]
    # his_array_size = case.train_data_range_days - case.predict_data_range_days + 1
    # move_length=int(case.predict_data_range_days/3)
    # move_length = 1
    # print 'move_length: ',move_length
    his_array_size = int((case.train_data_range_days- case.predict_data_range_days + 1)/move_length)
    if his_array_size<=input_num:
        his_array_size=2*input_num
        move_length=1
    # predict_size=4
    his_array = [0 for x in range(his_array_size)]
    # print 'his_array size: ',len(his_array)
    # print 'his_array: ',his_array
    #删除大的离谱的数
    sum=0.0
    max=0.0
    for x in range(len(case.his_data_all[flavor_num])):
        if(max<case.his_data_all[flavor_num][x]):
            max=case.his_data_all[flavor_num][x]
            max_number=x
        sum+=case.his_data_all[flavor_num][x]
    average=sum/len(case.his_data_all)
    # if(max>3*average and max>3):
    case.his_data_all[flavor_num][max_number]=4*average
    # print 'average: ',average
    move_ptr = 0
    # while move_ptr <= (case.train_data_range_days - case.predict_data_range_days):
    #     temp = 0
    #     for j in range(case.predict_data_range_days):
    #         tt=case.his_data_all[flavor_num][move_ptr + j]
    #         # if(tt>3*average and tt>2):
    #         #     tt-=3*average
    #         temp += tt
    #     # print 'move_ptr: ',move_ptr,'  temp: ',temp
    #     his_array[move_ptr / move_length] = temp
    #     move_ptr += move_length
    for i in range(his_array_size):
        temp = 0
        for j in range(case.predict_data_range_days):
            tt=case.his_data_all[flavor_num][move_ptr + j]
            # if(tt>3*average and tt>2):
            #     tt-=3*average
            temp += tt
        # print 'move_ptr: ',move_ptr,'  temp: ',temp
        his_array[i] = temp
        move_ptr += move_length
    # print "his_array处理", his_array
    #归一化处理
    mimi, maxi = getm(his_array)
    mimi = float(mimi)
    maxi = float(maxi)
    # print "max   ", maxi, "  mimi  ", mimi
    for flavor_num in range(len(his_array)):
        if(maxi-mimi<=0.1):
            his_array[flavor_num]=0.0
        else:
            his_array[flavor_num] = float((his_array[flavor_num] - mimi) / (maxi - mimi))
    # print "his_array", his_array
    # his_array = test_lof.ErrorDection(his_array, ave, de)
    train_data = [0 for x in range(len(his_array) - input_num)]
    # print 'his_array: '
    # print his_array
    for p in range(len(his_array) - input_num):
        # print 'p'
        # print p
        train_data[p] = his_array[p:p + input_num]
    # print 'train_data:',train_data
    test_data = [[0] for x in range(input_num, len(his_array))]
    for x in range(len(his_array) - input_num):
        # print 'x'
        # print x
        test_data[x][0] = his_array[x + input_num]
    # print 'test_data:', test_data
    return train_data, test_data, mimi, maxi,his_array

def get_train_data(case, flavor_num,move_length, input_num):
    # print 'flavor_num: ', flavor_num
    # print 'case.his_data_all[flavor_num]: ', case.his_data_all[flavor_num]
    his_array_size = 2*(case.train_data_range_days/case.predict_data_range_days) -1
    # move_length=int(case.predict_data_range_days/3)
    # move_length = 1
    # print 'move_length: ',move_length
    print 'his_array 1111 size: ', his_array_size
    # his_array_size = int(math.ceil((his_array_size + 0.0) / move_length))
    # predict_size=4
    his_array = [0 for x in range(his_array_size)]
    # print 'his_array size: ',len(his_array)
    # print 'his_array: ',his_array
    #删除大的离谱的数
    sum=0.0
    max=0.0
    for x in range(len(case.his_data_all[flavor_num])):
        if(max<case.his_data_all[flavor_num][x]):
            max=case.his_data_all[flavor_num][x]
            max_number=x
        sum+=case.his_data_all[flavor_num][x]
    average=sum/len(case.his_data_all)
    if(max>3*average and max>3):
        case.his_data_all[flavor_num][max_number]=3*average
    print 'average: ',average

    #更新his_array
    move_ptr = 0
    # while move_ptr <= (case.train_data_range_days-move_length):
    #     temp = 0
    #     for j in range(input_num):
    #         tt=case.his_data_all[flavor_num][move_ptr + j]
    #         # if(tt>3*average and tt>2):
    #         #     tt-=3*average
    #         temp += tt
    #     # print 'move_ptr: ',move_ptr,'  temp: ',temp
    #     his_array[move_ptr / move_length] = temp
    #     move_ptr += move_length
    for move_ptr in range(his_array_size):
        temp = 0
        for j in range(case.predict_data_range_days):
            print 'move_ptr*move_length: ',move_ptr*move_length
            tt=case.his_data_all[flavor_num][move_ptr*move_length + j]
            # if(tt>3*average and tt>2):
            #     tt-=3*average
            temp += tt
        print 'move_ptr: ',move_ptr,'  temp: ',temp
        his_array[move_ptr] = temp
        # move_ptr += move_length
    print "his_array处理", his_array
    #归一化处理
    mimi, maxi = getm(his_array)
    mimi = float(mimi)
    maxi = float(maxi)
    # print "max   ", maxi, "  mimi  ", mimi
    for flavor_num in range(len(his_array)):
        if(maxi-mimi<=0.1):
            his_array[flavor_num]=0.0
        else:
            his_array[flavor_num] = float((his_array[flavor_num] - mimi) / (maxi - mimi))
    print "归一化his_array", his_array
    # his_array = test_lof.ErrorDection(his_array, ave, de)
    train_data = [0 for x in range(his_array_size - input_num)]
    print 'train_data size: ',len(train_data)
    # print 'input_num: ',input_num
    # print his_array
    for p in range(len(his_array) - input_num):
        # print 'p'
        # print p
        train_data[p] = his_array[p:p + input_num]
    # print 'train_data:',train_data
    test_data = [[0] for x in range(his_array_size - input_num)]
    for x in range(len(his_array) - input_num):
        # print 'x'
        # print x
        test_data[x][0] = his_array[x + input_num]
    print 'test_data:', test_data
    return train_data, test_data, mimi, maxi,his_array

def getm(A):
    mini=10000.0
    maxi=0.0
    for i in range(len(A)):
        if mini>A[i]:
            mini=A[i]
        elif maxi<A[i]:
            maxi=A[i]
    return mini,maxi

def print_dict_flavor(dict):
    for x in dict:
        print dict[x].name+' mem:'+str(dict[x].mem)+' cpu:',str(dict[x].cpu),


