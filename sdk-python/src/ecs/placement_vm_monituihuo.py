#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-12 10:30:54
# @Author  : mianhk (yugc666@163.com)
# @Link    : ${link}
# @Version : $Id$


import random
import math
import Tools
import case_info



def put_flavors_to_servers(map_predict_num_flavors, map_flavor_cpu_mem,server_mem,server_cpu,CPUorMEM,result):
    # predict_flavor_dict, case, result
    flavors_array = []
    # print 'map_predict_num_flavors: ',map_predict_num_flavors
    #将预测出来的所有虚拟机都加入到vec_flavors中
    for i in map_predict_num_flavors:
        while (map_predict_num_flavors[i]>0):
            map_predict_num_flavors[i]-=1
            # print 'map_flavor_cpu_mem[i]: ',map_flavor_cpu_mem[i]
            flavors_array.append(map_flavor_cpu_mem[i])

    # print 'vec_flavors'
    # for x in vec_flavors:
    #     print x.name
    min_server=len(flavors_array)+1
    res_servers=[] #用于存放最优结果
    T=350
    Tmin=1
    r=0.999
    random_dice=[]

    for i in range(len(flavors_array)):
        random_dice.append(i)
    while(T>Tmin):
        random.shuffle(random_dice)
        new_flavors_array=flavors_array
        new_flavors_array[random_dice[0]], new_flavors_array[random_dice[1]]=new_flavors_array[random_dice[1]], new_flavors_array[random_dice[0]]

        servers=[]
        firstserver=case_info.Server(server_mem,server_cpu)
        # print 'len(firstserver.flavors):   ',len(firstserver.flavors)
        servers.append(firstserver)

        for i in range(len(new_flavors_array)):
            flag=0
            for x in range(len(servers)):
                if (servers[x].put_flavor(new_flavors_array[i]) == True):
                    flag=1
                    break
            if(flag==0):
                newserver = case_info.Server(server_mem,server_cpu)
                newserver.put_flavor(new_flavors_array[i])
                servers.append(newserver)

        if(CPUorMEM=='CPU'):
            server_num=len(servers)-1+servers[-1].get_cpu_usage_rate()
        else:
            server_num = len(servers) - 1 + servers[-1].get_mem_usage_rate()

        if (server_num < min_server):
            min_server = server_num
            res_servers = servers
            flavors_array = new_flavors_array
        else:
            # if math.exp((min_server - server_num) / T) >random.random():#  random.random()/ RAND_MAX
            if 1 /(1+math.exp((server_num-min_server) / T)) >random.random():#  random.random()/ RAND_MAX
                min_server = server_num
                res_servers = servers
                flavors_array = new_flavors_array
        T = r * T # 一次循环结束，温度降低
    #打印资源利用率
    # res_resources_cpu=0
    # res_resources_mem=0
    # res_total_cpu=0
    # res_total_mem=0
    # for x in servers:
    #     res_resources_cpu+=x.free_cpu
    #     res_resources_mem+=x.free_mem
    #     res_total_cpu+=x.total_cpu
    #     res_total_mem+=x.total_mem
    # print '资源利用率为： cpu利用',1-(res_resources_cpu+0.0)/res_total_cpu,'  mem利用率为:   ',1-(res_resources_mem+0.0)/res_total_mem
    #结果处理
    result.append(len(res_servers))
    for x in range(len(res_servers)):
        temp = {}
        for i in res_servers[x].flavors:
            if(Tools.isContainKey(temp,i.name)):
                temp[i.name]+=1
            else:
                temp[i.name] = 1
        temp_result=str(x+1)+' '
        for j in temp:
            temp_result+=j+' '+str(temp[j])+' '
        result.append(temp_result)

    return result

def calcu_vm(predict_flavor_dict, case, result):
    #计算预测的虚拟机的数目
    sum_of_predict_flavor = 0
    result_tmp = ''
    for i in predict_flavor_dict:
        # print 'predict_flavor_dict[i]:  '+str(predict_flavor_dict[i])
        sum_of_predict_flavor += predict_flavor_dict[i]
        result_tmp += str(i) + ' ' + str(predict_flavor_dict[i]) + '\r\n'
    # print 'sum_of_predict_flavor:'+str(sum_of_predict_flavor)
    result.append(int(sum_of_predict_flavor))
    result.append(result_tmp)

    # 计算物理服务器的个数和分布
    result=put_flavors_to_servers(predict_flavor_dict, case.flavors_type, case.MEM, case.CPU, case.opt_target,result)
    return result
