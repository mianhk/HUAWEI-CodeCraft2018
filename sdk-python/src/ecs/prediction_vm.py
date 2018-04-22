#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-11 10:19:20
# @Author  : mianhk (yugc666@163.com)
# @Link    : ${link}
# @Version : $Id$


import math
import random

import case_info
import Tools
import lstm_main

def predict_all(case):
    # print 'case.his_data'
    predict_flavor_dict={}  #新建预测字典
    #对于每一个his_data训练数据中的虚拟机遍历
    for flavor in case.his_data:
        #如果该虚拟机在预测的虚拟机列表中，则开始进行预测
        if (flavor in case.flavors_type):
            flavor_num=case.flavors_type[flavor].name # flavor的name
            # predict_flavor_dict[i] = int(randomforest.RandomForestRegression(case, i))
            predict_flavor_dict=e_prediction_call(case, flavor_num, predict_flavor_dict)  #调用predict_one预测每一台虚拟机
            # print 'case.flavors_type[flavor].name: ',flavor_num
            # print 'case.his_data_all[flavor_num] ',case.his_data_all[flavor_num]
            #lstm预测
            # 结点数设置
            # if(case.predict_data_range_days>7):
            #     input_num = 7
            #     move_length = 3
            # else:
            #     input_num = case.predict_data_range_days
            #     move_length = case.predict_data_range_days / 2
            # cell_num = input_num+1
            # move_length = case.predict_data_range_days / 2
            # print 'move_length: ', move_length
            # predict_flavor_dict[flavor_num] = int(lstm_main.lstm_main(case,flavor_num,input_num,cell_num,move_length))
            # print "predict_flavor_dict[i]", predict_flavor_dict[flavor_num]
    return predict_flavor_dict

#指数平滑法预测
def e_prediction_call(case, flavor_num, predict_flavor_dict):
    '''
    描述：预测单个虚拟机
    :param case:  输入的case
    :param flavor_num: 第几个虚拟机
    :param predict_flavor_dict: 输出字典
    :return: predict_flavor_dict
    '''
    train_data_range_days=case.train_data_range_days/Tools.TRAIN_GRAIN_DAYS #训练时间的周期=总的训练天数/训练的时间粒度
    predict_data_range_days=case.predict_data_range_days/Tools.TRAIN_GRAIN_DAYS#预测时间的周期=总的预测天数/训练的时间粒度
    whole_days=train_data_range_days+predict_data_range_days #总的周期
    # whole_days=case.train_data_range_days+case.predict_data_range_days #总的周期
    # predict_flavor_dict=[]
    his_array=[0 for x in range(whole_days)] #历史数组初始化
    ave=0
    max=0
    max_num=0
    # print 'case.his_data_all[flavor_num]: ', case.his_data_all[flavor_num]
    # 删除大的离谱的数
    sum = 0.0
    max = 0.0
    for x in range(len(case.his_data_all[flavor_num])):
        if (max < case.his_data_all[flavor_num][x]):
            max = case.his_data_all[flavor_num][x]
            max_number = x
        sum += case.his_data_all[flavor_num][x]
    average = sum / len(case.his_data_all)
    if (max > 3 * average and max > 3):
        case.his_data_all[flavor_num][max_number] = int(round(1 * average))
    # print 'case.his_data_all[flavor_num]: ',case.his_data_all[flavor_num]
    for p in range(len(case.his_data_all[flavor_num])):
        his_array[p/Tools.TRAIN_GRAIN_DAYS]+=case.his_data_all[flavor_num][p]
    #调用指数平滑预测,两种不同，有bug
    # predict_flavor_dict[i]=int(predict_array1[train_data_range_days])
    # ave=remove_noise.average(his_array)
    # delat=remove_noise.delat(his_array,ave)
    # his_array=remove_noise.remove_noise(his_array)
        predict_flavor_dict[flavor_num]= int(e_prediction(his_array, train_data_range_days, predict_data_range_days, 3)) + 10
    # predict_flavor_dict[i]=int(predict_flavor_dict[i]*delat+ave)
    # predict_flavor_dict[i]=int(e_prediction(his_array,train_data_range_days,predict_data_range_days, 3))
    # print 'predict_flavor_dict[i]:  '
    # print predict_flavor_dict[i]
    return predict_flavor_dict


def random_predict(case, i, predict_flavor_dict):
    total_length=case.train_data_range_days
    his_array=[0 for x in range(total_length)]
    for x in range(case.train_data_range_days):
        if(Tools.isContainKey(case.his_data[i],x)):
            his_array[x]=case.his_data[i][x]
    # print 'his_array: ',his_array
    predict_flavor_dict[i]=0

    for t in range(total_length-case.predict_data_range_days,total_length):
        # print 'his_array[t]: ',his_array[t]
        # predict_flavor_dict[i]+=int(his_array[t]*(random.random()+0.6))
        predict_flavor_dict[i]+=int(his_array[int(t*(random.random()))])
    # print 'predict_flavor_dict[i]: ', i,'  ',predict_flavor_dict[i]
    return predict_flavor_dict


def e_prediction(his_array,train_data_range_days,predict_data_range_days,times):
    # his_array = test_lof.ErrorDection(his_array)
    a1 = 0.962 # 加权系数a  0.026最高分的一个了。。
    # a1 = 0.962 # 加权系数a  0.026最高分的一个了。。
    # print "now   hahah",his_array

    predict_array1=his_array
    # y=predict_array1
    y2=predict_array1
    y3=predict_array1
    #一次平滑
    # 一阶
    # 确定第一个
    if (len(predict_array1) >= 3):
        predict_array1[0]=(his_array[0]+his_array[1]+his_array[2]+0.0)/3
    else:
        predict_array1[0] = his_array[0]
    #一阶循环
    for j in range(1, len(predict_array1)):
        predict_array1[j] = (a1) * his_array[j] + (1-a1) * predict_array1[j - 1]
    # 二阶
    predict_array2 = predict_array1
    #二阶第一个
    if (len(predict_array1) >= 3):
        predict_array2[0] = (predict_array1[0] + predict_array1[1] + predict_array1[2] + 0.0) / 3
    else:
        predict_array2[0] = predict_array1[0]
    #二阶循环
    for j in range(1, len(predict_array1)):
        predict_array2[j] = (a1) * predict_array1[j] + (1-a1) * predict_array2[j - 1]
        at2=2*(predict_array1[j])-predict_array2[j]
        bt2=a1/(1-a1)*(predict_array1[j]-predict_array2[j])
        y2[j] = at2 + bt2 * j

    res=0
    for x in range(predict_data_range_days):
            res+=y2[-x-1]
    return res
    # return y3[-1]
    # return  predict_array1





