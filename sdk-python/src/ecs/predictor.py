#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-10 22:43:06
# @Author  : mianhk (yugc666@163.com)
# @Link    : ${link}
# @Version : $Id$
import case_info
import prediction_vm
import placement_vm_monituihuo


def predict_vm(ecs_lines, input_lines):
    # Do your work from here#
    result = []
    if ecs_lines is None:
        print 'ecs information is none'
        return result
    if input_lines is None:
        print 'input file information is none'
        return result

    # 获取case中的数据
    case = case_info.Case(ecs_lines, input_lines)
    print 'case.train_data_range_days: ',case.train_data_range_days
    print 'case.predict_data_range_days: ',case.predict_data_range_days
    # 调用预测函数，返回预测后的字典：结构为{'flavor3': 0, 'flavor2': 2...}每一个虚拟机及其对应的个数
    # print 'case.his_data: ',case.his_data
    #普通预测
    predict_flavor_dict = prediction_vm.predict_all(case)
    # 调用放置虚拟机函数，将预测的虚拟机放在物理机上
    # print predict_flavor_dict
    #自适应放置算法
    # result = placement_vm.calcu_vm(predict_flavor_dict, case, result)
    #模拟退火放置
    result=placement_vm_monituihuo.calcu_vm(predict_flavor_dict, case, result)
    return result
