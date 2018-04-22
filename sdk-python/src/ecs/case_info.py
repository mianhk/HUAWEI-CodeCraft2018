#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-10 22:44:22
# @Author  : mianhk (yugc666@163.com)
# @Link    : ${link}
# @Version : $Id$





import collections
import Tools
# 虚拟机类
class Flavor(object):
    name=''
    mem=0
    cpu=0

    def __init__(self,name,mem,cpu):
        self.name=name
        self.mem = mem
        self.cpu=cpu
#服务器类
class Server(object):
    total_mem = 0
    total_cpu=0
    free_mem = 0  # 初始化时剩余内存等于总内存
    free_cpu=0  #初始化时剩余CPU等于总CPU

    flavors=[] #物理服务器已存放虚拟机列表

    def __init__(self,mem,cpu):
        self.total_cpu=cpu
        self.total_mem=mem
        self.free_cpu=cpu
        self.free_mem=mem
        self.flavors=[]

    def put_flavor(self,flavor):
        if (self.free_mem >= flavor.mem and self.free_cpu >= flavor.cpu ):
            self.free_cpu -= flavor.cpu
            self.free_mem -= flavor.mem
            self.flavors.append(flavor)
            return True
        return False
    #获取服务器CPU使用率
    def get_cpu_usage_rate(self):
        return self.free_cpu/(self.total_cpu)
    #获取服务器内存使用率
    def get_mem_usage_rate(self):
        return self.free_mem / (self.total_mem)

class Case(object):
    '''
    输入文件的信息
    '''
    CPU = 0  # 每台物理机的cpu数
    MEM = 0  # 每台物理机的内存大小 单位G

    opt_target = ''  # 优化目标，值为CPU和MEM

    flavors_type = []  # {虚拟机：[CPU，MEM]} 虚拟机类型,字典：eg:{'flavor3': [1, 4], 'flavor2': [1, 2]}

    time_grain = -1  # 预测时间粒度
    date_range_size = 0  # 需要预测的时间量
    train_data_range = []  # 预测开始时间与结束时间，左闭右开

    train_data_range_days = 0  # 训练数据的天数
    predict_data_range_days = 0  # 预测的天数，简单处理

    # 训练数据中虚拟机、日期二维映射表 [虚拟机类型,日期]
    his_data = {}
    # 训练数据中虚拟机、和对应的
    his_data_all={}

    #case的初始化
    def __init__(self, origin_train_data, origin_case_info, predict_time_grain=Tools.TIME_GRAIN_DAY):
        self.time_grain = predict_time_grain   #修改时间粒度
        self.set_case_info(origin_case_info, predict_time_grain)  #获取input文件的数据
        self.set_his_data(origin_train_data, predict_time_grain) #获取训练数据

    def set_case_info(self, origin_case_info, predict_time_grain):
        # 处理 CPU MEM HDD
        temp = origin_case_info[0].split(' ')
        self.CPU = int(temp[0])
        self.MEM = int(temp[1])

        # 处理虚拟机类型
        tsize = int(origin_case_info[2].replace('\r', ''))
        self.flavors_type = {}
        for i in range(tsize):
            _type = origin_case_info[3 + i].replace('\r', '')
            typename, cpu, mem = _type.split(' ')
            temp_flavors=Flavor(typename,int(mem)/1024,int(cpu))
            # self.flavors_type[temp_flavors.name] = [int(_cpu), int(_mem) / 1024]
            self.flavors_type[temp_flavors.name] = temp_flavors
        # Tools.print_dict_flavor(self.flavors_type)

        # 处理优化目标
        self.opt_target = origin_case_info[4 + tsize][0:3]

        # 需要预测时间处理
        _st = origin_case_info[6 + tsize]
        _et = origin_case_info[7 + tsize]
        self.predict_data_range_days=Tools.calcu_days(_st,_et)
        # print 'self.data_range_days'  # 打印预测的天数
        # print self.data_range_days

    def set_his_data(self, origin_train_data, predict_time_grain):
        hisdata = {}
        train_begin_time = origin_train_data[0].split('\t')[2]  #训练时间开始
        train_end_time = origin_train_data[-1].split('\t')[2]   #训练时间结束
        self.train_data_range_days=Tools.calcu_days(train_begin_time,train_end_time)+1  #计算训练时间的天数
        # print 'train_data_range_days: ',self.train_data_range_days
        for line in origin_train_data:
            _, vm_type, time = line.split('\t')
            if not Tools.isContainKey(hisdata, vm_type):
                hisdata[vm_type] = collections.OrderedDict()
            ##按天数存储
            day_index = Tools.calcu_days(train_begin_time,Tools.get_grain_time(time, predict_time_grain))+1 #加1表示从第一天开始
            #除以训练的时间粒度
            # gt=gt/Tools.TRAIN_GRAIN_DAYS+1  #加1天，表示从1开始
            # print '*******gt:    '
            # print day_index
            point = hisdata[vm_type]
            if not Tools.isContainKey(point, day_index):
                point[day_index] = 0
            cot = point[day_index] + 1
            point[day_index] = cot
        self.his_data = hisdata

        for i in self.his_data:
            his_array=[0 for x in range(self.train_data_range_days)]
            for x in range(self.train_data_range_days):
                if (Tools.isContainKey(self.his_data[i], x)):
                    his_array[x]=self.his_data[i][x]
            self.his_data_all[i]=his_array
        # print 'self.his_data_all: ',self.his_data_all








