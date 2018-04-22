#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-10 22:44:22
# @Author  : mianhk (yugc666@163.com)
# @Link    : ${link}
# @Version : $Id$

'''
程序说明
%  1、数据为7周，每天的虚拟机数量，用前三周推测第四周训练，依次类推。最后用训练出来的权重矩阵0 14 4 2 5 4 3 2
%  2、LSTM网络输入结点为3，输出结点为1个，隐藏结点4个
'''
import Tools
import random
import sys



import math


INT_RANDOM=0.8
INT_RANDOMA=0.8

def LSTM_updata_weight(input_num,cell_num,output_num,n, yita, Error,weight_input_x,weight_input_h,weight_inputgate_x,weight_inputgate_c,weight_forgetgate_x,weight_forgetgate_c,weight_outputgate_x,weight_outputgate_c,weight_preh_h,cell_state, h_state,input_gate, forget_gate,output_gate, gate,train_data, pre_h_state,input_gate_input,output_gate_input,forget_gate_input):
    data_length=input_num

    cur_cell_state=matiix_get_col(cell_state, n)
    pre_cell_state=matiix_get_col(cell_state, n-1)
    ones_matrix=[1 for i in range(len(cur_cell_state))]
    cur_train_data = matiix_get_col(train_data, n)
    weight_preh_h_temp = weight_preh_h
    reverse_weight_preh_h=matrix_get_colTOrow(weight_preh_h)
    # print '看一下ERROR：',Error
    Error=2*Error
    yita=0.1

    yita = 0.1
    yita_weight_pre_h = 0.05
    yita_weight_outputgate_x = 0.05
    yita_weight_inputgate_x = 0.05
    yita_weight_input_x = 0.11  # 小 于0.1分数会变小 0.12-78.824
    yita_weight_forgetgate_x = 0.08
    yita_weight_inputgate_c = 0.4
    yita_weight_forgetgate_c = 0.05
    yita_weight_outputgate_c = 0.5  # 0.05分数变低
    yita_weight_input_h = yita

    #更新weight_pre_h的权重
    delta_weight_preh_h_tmp=[([0] * output_num) for i in range(cell_num)]
    mrtiix_col_alter(delta_weight_preh_h_tmp, 0, matrix_mul_numSingleA(pre_h_state,Error))
    weight_preh_h_temp=matrixSubColSingle(weight_preh_h_temp,matrix_mul_num(delta_weight_preh_h_tmp,yita_weight_pre_h))

    #更新weight_outputgate_x的权重
    delta_weight_outputgate_x=[([0] * cell_num) for i in range(input_num)]

    for m in range(data_length):
        tmp1=matrix_dot_mulSingle(matrix_mul_numSingleA(reverse_weight_preh_h,Error),matrix_tanhSingA(cur_cell_state))
        tmp2=matrix_dot_mulSingle(matrix_expSingleA(matrix_mul_numSingleA(output_gate_input,-1)),matrix_mul_numSingleA(matrix_SquareSingleA(output_gate),train_data[m][n]))
        mrtiix_row_alter(delta_weight_outputgate_x, m, matrix_dot_mulSingle(tmp1,tmp2))
    weight_outputgate_x = matrixSub(weight_outputgate_x , matrix_mul_num(delta_weight_outputgate_x,yita_weight_outputgate_x))

    #更新weight_inputgate_x的权重
    delta_weight_inputgate_x = [([0] * cell_num) for i in range(input_num)]
    for m in range(data_length):
        tmp1=matrix_dot_mulSingle(matrix_mul_numSingleA(reverse_weight_preh_h,Error),output_gate)
        tmp2=matrix_dot_mulSingle(matrixSubSingle(ones_matrix,matrix_SquareSingleA(matrix_tanhSingA(cur_cell_state))),gate)
        tmp3=matrix_dot_mulSingle(matrix_expSingleA(matrix_mul_numSingleA(input_gate_input,-1)),matrix_mul_numSingleA(matrix_SquareSingleA(input_gate),train_data[m][n]))
        mrtiix_row_alter(delta_weight_inputgate_x, m, matrix_dot_mulSingle(matrix_dot_mulSingle(tmp1,tmp2),tmp3))
    weight_inputgate_x = matrixSub(weight_inputgate_x , matrix_mul_num(delta_weight_inputgate_x,yita_weight_input_x))


    if n!=0:
        #更新weight_input_x
        delta_weight_input_x= [([0] * cell_num) for i in range(input_num)]
        temp=matrixPlusSingle(matrixMulSingleA(cur_train_data,weight_input_x), matrix_mul_numSingleA(weight_input_h,h_state[n-1]))
        one_temp_matrix=[1 for i in range(len(temp))]
        for m in range(input_num):
            tmp1 = matrix_dot_mulSingle(matrix_mul_numSingleA(reverse_weight_preh_h, Error), output_gate)
            tmp2 = matrix_dot_mulSingle(matrixSubSingle(ones_matrix, matrix_SquareSingleA(matrix_tanhSingA(cur_cell_state))), input_gate)
            tmp3 =matrix_mul_numSingleA(matrixSubSingle(one_temp_matrix, matrix_SquareSingleA(matrix_tanhSingA(temp))), train_data[m][n])
            mrtiix_row_alter(delta_weight_input_x, m,matrix_dot_mulSingle(matrix_dot_mulSingle(tmp1, tmp2), tmp3))
        weight_input_x=matrixSub(weight_input_x, matrix_mul_num(delta_weight_input_x, yita_weight_inputgate_x))

        # 更新weight_forgetgate_x的权重
        delta_weight_forgetgate_x = [([0] * cell_num) for i in range(input_num)]
        for m in range(data_length):
            tmp1 = matrix_dot_mulSingle(matrix_mul_numSingleA(reverse_weight_preh_h, Error), output_gate)
            tmp2 = matrix_dot_mulSingle(matrixSubSingle(ones_matrix, matrix_SquareSingleA(matrix_tanhSingA(cur_cell_state))), pre_cell_state)
            tmp3 = matrix_dot_mulSingle(matrix_expSingleA(matrix_mul_numSingleA(forget_gate_input, -1)),matrix_mul_numSingleA(matrix_SquareSingleA(forget_gate), train_data[m][n]))
            mrtiix_row_alter(delta_weight_forgetgate_x, m, matrix_dot_mulSingle(matrix_dot_mulSingle(tmp1, tmp2), tmp3))
            weight_forgetgate_x = matrixSub(weight_forgetgate_x, matrix_mul_num(delta_weight_forgetgate_x, yita_weight_forgetgate_x))

        # 更新weight_inputgate_c的权重
        delta_weight_inputgate_c = [([0] * cell_num) for i in range(cell_num)]
        for m in range(cell_num):
            tmp1 = matrix_dot_mulSingle(matrix_mul_numSingleA(reverse_weight_preh_h, Error), output_gate)
            tmp2 = matrix_dot_mulSingle(matrixSubSingle(ones_matrix, matrix_SquareSingleA(matrix_tanhSingA(cur_cell_state))), gate)
            tmp3 = matrix_dot_mulSingle(matrix_expSingleA(matrix_mul_numSingleA(input_gate_input, -1)),matrix_mul_numSingleA(matrix_SquareSingleA(input_gate), cell_state[m][n-1]))
            mrtiix_row_alter(delta_weight_inputgate_c, m, matrix_dot_mulSingle(matrix_dot_mulSingle(tmp1, tmp2), tmp3))
        weight_inputgate_c = matrixSub(weight_inputgate_c, matrix_mul_num(delta_weight_inputgate_c, yita_weight_inputgate_c))

        # 更新weight_forgetgate_c的权重
        delta_weight_forgetgate_c= [([0] * cell_num) for i in range(cell_num)]
        for m in range(cell_num):
            tmp1 = matrix_dot_mulSingle(matrix_mul_numSingleA(reverse_weight_preh_h, Error), output_gate)
            tmp2 = matrix_dot_mulSingle(matrixSubSingle(ones_matrix, matrix_SquareSingleA(matrix_tanhSingA(cur_cell_state))), pre_cell_state)
            tmp3 = matrix_dot_mulSingle(matrix_expSingleA(matrix_mul_numSingleA(forget_gate_input, -1)),matrix_mul_numSingleA(matrix_SquareSingleA(forget_gate), cell_state[m][n-1]))
            mrtiix_row_alter(delta_weight_forgetgate_c, m, matrix_dot_mulSingle(matrix_dot_mulSingle(tmp1, tmp2), tmp3))
        weight_forgetgate_c = matrixSub(weight_forgetgate_c, matrix_mul_num(delta_weight_forgetgate_c, yita_weight_forgetgate_c))

        # 更新weight_outputgate_c的权重
        delta_weight_outputgate_c = [([0] * cell_num) for i in range(cell_num)]
        for m in range(cell_num):
            tmp1 = matrix_dot_mulSingle(matrix_mul_numSingleA(reverse_weight_preh_h, Error), matrix_tanhSingA(cur_cell_state))
            tmp3 = matrix_dot_mulSingle(matrix_expSingleA(matrix_mul_numSingleA(output_gate_input, -1)),matrix_mul_numSingleA(matrix_SquareSingleA(output_gate), cell_state[m][n-1]))
            mrtiix_row_alter(delta_weight_outputgate_c, m,matrix_dot_mulSingle(tmp1, tmp3))
        weight_outputgate_c = matrixSub(weight_outputgate_c, matrix_mul_num(delta_weight_outputgate_c, yita_weight_outputgate_c))

        #更新weight_input_h
        delta_weight_input_h= [0  for i in range(cell_num)]

        temp=matrixPlusSingle(matrixMulSingleA(cur_train_data,weight_input_x), matrix_mul_numSingleA(weight_input_h,h_state[n-1]))
        one_temp_matrix=[1 for i in range(len(temp))]
        for m in range(output_num):
            tmp1 = matrix_dot_mulSingle(matrix_mul_numSingleA(reverse_weight_preh_h, Error), output_gate)
            tmp2 = matrix_dot_mulSingle(matrixSubSingle(ones_matrix, matrix_SquareSingleA(matrix_tanhSingA(cur_cell_state))), input_gate)
            tmp3 =matrix_mul_numSingleA(matrixSubSingle(one_temp_matrix, matrix_SquareSingleA(matrix_tanhSingA(temp))), h_state[n-1])
            mrtiix_row_alter(delta_weight_forgetgate_x, m,matrix_dot_mulSingle(matrix_dot_mulSingle(tmp1, tmp2), tmp3))
        weight_input_h=matrixSubSingle(weight_input_h, matrix_mul_numSingleA(delta_weight_input_h, yita_weight_input_h))

    else:
        #更新weight_input_x
        delta_weight_input_x= [([0] * cell_num) for i in range(input_num)]
        temp=matrixMulSingleA(cur_train_data,weight_input_x)
        one_temp_matrix=[1 for i in range(len(temp))]
        for m in range(input_num):
            tmp1 = matrix_dot_mulSingle(matrix_mul_numSingleA(reverse_weight_preh_h, Error), output_gate)
            tmp2 = matrix_dot_mulSingle(matrixSubSingle(ones_matrix, matrix_SquareSingleA(matrix_tanhSingA(cur_cell_state))), input_gate)
            tmp3 =matrix_mul_numSingleA(matrixSubSingle(one_temp_matrix, matrix_SquareSingleA(matrix_tanhSingA(temp))), train_data[m][n])
            mrtiix_row_alter(delta_weight_input_x, m,matrix_dot_mulSingle(matrix_dot_mulSingle(tmp1, tmp2), tmp3))

        weight_input_x=matrixSub(weight_input_x, matrix_mul_num(delta_weight_input_x, yita_weight_input_x))
    weight_preh_h = weight_preh_h_temp
    return weight_input_x, weight_input_h,weight_inputgate_x, weight_inputgate_c,weight_forgetgate_x, weight_forgetgate_c,weight_outputgate_x,weight_outputgate_c,weight_preh_h



def init_matrix_random(row, col,div):
    res=[([0] * col) for i in range(row)]
    # print res
    for i in range(row):
        for j in range(col):
            # res[i][j]=random.random()/div
            res[i][j]=INT_RANDOM/div
            # print res
    return res
#初始化一维数组
def init_matrix_randomA(col, div):
    res=[0 for i in range(col)]
    for i in range(col):
        # res[i]=random.random()/div
        res[i]=INT_RANDOMA/div
            # print res
    return res

def matrixMul(A, B):
    C=[([0] * len(A)) for i in range(len(B[0]))]
    #res = [[0] * len(B[0]) for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] = C[i][j] +A[i][k] * B[k][j]
    return C

#A是一维数组相乘的情况
def matrixMulSingleA(A, B):
    C = [0 for t in range(len(B[0]))]
    # res = [[0] * len(B[0]) for i in range(len(A))]
    for i in range(len(B[0])):
        for j in range(len(A)):
            C[i] = C[i] + A[j] * B[j][i]
    return C
#Ab都是一维数组相乘的情况
def matrixMulSingleAB(A, B):
    C=0
    for j in range(len(A)):

        C = C+ A[j] * B[j][0]

    return C
def matrix_mul_num(A, num):
    C=A
    for i in range(len(A)):
        for j in range(len(A[0])):
            # print 'i =  '+str(i)+'  j= '+str(j)
            # print 'A[i][j]'
            # print A[i][j]
            C[i][j]=A[i][j]*num
            # print 'A[i][j]'
            # print A[i][j]
    return C
#矩阵的某一列乘以一个数
def mrtiix_col_mul_num(A,i,num):
    C=A
    for x in range(len(A[0])):
        # print "C"
        # print C[x][i]
        C[x][i]=(A[x][i])*num
    return C

def mrtiix_col_mul_num(A,i,num):
    C=A
    for x in range(len(A[0])):
        # print "C"
        # print C[x][i]
        C[x][i]=(A[x][i])*num
    return C


def mrtiix_col_mul_row(A,i,B):
    C=A
    for x in range(len(A[0])):
        # print "C"
        # print C[x][i]
        C[x][i]=(A[x][i])*B[i][x]
    return C
#h获得矩阵的某一列
def matiix_get_col(A,i):

    tmp = [0 for t in range(len(A))]
    for x in range(len(A)):
         tmp[x]=A[x][i]
    return tmp
#更新矩阵的每一列
def mrtiix_col_alter(A,i,B):
    for x in range(len(A)):
        A[x][i]=B[x]
    return A
#更新矩阵的每一行
def mrtiix_row_alter(A,i,B):
    for x in range(len(B)):

        A[i][x]=B[x]
    return A

def matrix_reverse(A):
    matrix=A
    return [[row[col] for row in matrix] for col in range(len(matrix[0]))]

#行转为列
def matrix_reverse_one(A):
    matrix = [([0] * len(A)) * 1]
    for i in range(len(A)):
        matrix[0][i] = A[i]
    return matrix
#矩阵exp
def matrix_expSingleA(A):
    for i in range(len(A)):
        A[i]=math.exp(A[i])
            # print 'A[i][j]'
            # print A[i][j]
    return A
#矩阵tanh
def matrix_tanhSingA(A):
    C  = [0 for t in range(len(A))]
    for i in range(len(A)):
            C[i]=math.tanh(A[i])
    return C
#矩阵^2
def matrix_SquareSingleA(A):
    res = [0 for t in range(len(A))]
    for i in range(len(A)):
        res[i]=A[i]*A[i]
    return res


#实现矩阵减法

def matrixSub(A, B):
    res = [[0] * len(A[0]) for i in range(len(A))]
    for i in range(len(A)):
             for k in range(len(A[0])):
                res[i][k] += A[i][k]- B[i][k]
    return res

#实现矩阵加法
def matrixPlus(A, B):
    res = [[0] * len(A[0]) for i in range(len(A))]
    for i in range(len(A)):
             for k in range(len(A[0])):
                res[i][k] += A[i][k]+B[i][k]
    return res
#A,B为一维数组相加的情况
def matrixPlusSingle(A, B):
    res = [0 for t in range(len(A))]
    for i in range(len(A)):
             res[i]= A[i]+B[i]
    return res
#A,B为一维数组相sub的情况
def matrixSubSingle(A, B):
    res = [0 for t in range(len(A))]
    for i in range(len(A)):
             res[i]= A[i]-B[i]
    return res
def matrixSubColSingle(A, B):
    res = [([0] * len(A[0])) for t in range(len(A))]
    for i in range(len(A)):
             res[i][0]= A[i][0]-B[i][0]
    return res
def matrixSubSingle(A, B):
    res = [0 for t in range(len(A))]
    for i in range(len(A)):
             res[i]= A[i]-B[i]
    return res
def matrix_sum(A):
    sum=0
    for i in range(len(A)):
             sum+=A[i]*A[i]
    return sum
#实现矩阵点乘
def matrix_dot_mul(A,B):
    matrix=[([0] * len(A)) for i in range(len(B[0]))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            matrix[i][j]=A[i][j]*B[i][j]
    return matrix
#实现一维矩阵点乘
def matrix_dot_mulSingle(A,B):
    matrix=[0 for i in range(len(A))]
    for i in range(len(A)):
         matrix[i]=A[i]*B[i]
    return matrix
def matrix_get_colTOrow(A):
    matrix=[0 for i in range(len(A))]
    for i in range(len(A)):
         matrix[i]=A[i][0]
    return matrix

def matrix_mul_num(A, num):
    for i in range(len(A)):
        for j in range(len(A[0])):
            # print 'i =  '+str(i)+'  j= '+str(j)
            # print 'A[i][j]'
            # print A[i][j]
            A[i][j]=A[i][j]*num
            # print 'A[i][j]'
            # print A[i][j]
    return A
# 一维矩阵乘常数
def matrix_mul_numSingleA(A, num):
    matrix = matrix = [0 for i in range(len(A))]
    for i in range(len(A)):
        matrix[i]=A[i]*num
    return matrix
def average(A):
    averageh=0.0
    for i in range(len(A)):
        averageh+=A[i]
    averageh=averageh/len(A)
    return averageh
def delat(A,averagehi):
    delta=0.0
    for i in range(len(A)):
        delta+=(A[i]-averagehi)*(A[i]-averagehi)
    delta=delta/len(A)
    return delta
def temp_update(temp,h_state_final):
    for i in range(len(temp)-1):
        temp[i]=temp[i+1]
    temp[len(temp)-1]=h_state_final
    return temp
def getm(A):
    mini=10000.0
    maxi=0.0
    for i in range(len(A)):
        if mini>A[i]:
            mini=A[i]
        elif maxi<A[i]:
            maxi=A[i]
    return mini,maxi


def lstm_main(case, flavor_num,input_num,cell_num,move_length = 1):
    # 初始化h_state_final
    output_num = 1
    # 数据加载，并归一化处理

    # train_data, test_data ,mimi, maxi= LSTM_data_process(case, flavor_num, move_length,input_num)
    train_data, test_data ,mimi, maxi,his_array= Tools.get_train_data2(case, flavor_num, move_length,input_num)
    # print 'train_data: ',train_data
    train_data = matrix_reverse(train_data)
    test_data = matrix_reverse(test_data)
    data_num = len(train_data[0])

    #网络参数初始化

    #网络中门的偏置  分别初始化输入门、遗忘门、输出门权重共8个矩阵
    aa=1
    bias_input_gate = init_matrix_randomA(cell_num, aa)
    bias_forget_gate = init_matrix_randomA( cell_num, aa)
    bias_output_gate = init_matrix_randomA(cell_num, aa)
    #网络权重初始化
    ab = 20
    ac=40
    weight_input_x = init_matrix_random(input_num, cell_num,ab)
    weight_input_h = init_matrix_randomA(cell_num, ac)  #如果修改了输出要注意这个地方
    weight_inputgate_x = init_matrix_random(input_num, cell_num,ab)
    weight_inputgate_c = init_matrix_random(cell_num, cell_num,  ab)
    weight_forgetgate_x = init_matrix_random(input_num, cell_num, ab)
    weight_forgetgate_c = init_matrix_random(cell_num, cell_num, ab)
    weight_outputgate_x = init_matrix_random(input_num, cell_num, ab)
    weight_outputgate_c = init_matrix_random(cell_num, cell_num, ab)
    # hidden_output权重  初始化隐藏层到输出的权重矩阵
    weight_preh_h =init_matrix_random(cell_num, output_num, 1)
    # 网络状态初始化  初始化隐藏层状态和细胞状态
    cost_gate = 0.1
    # print cost_gate
    ad=1
    ae=1
    h_state =init_matrix_randomA( data_num, ad)
    cell_state=init_matrix_random(cell_num, data_num, ae)

    ################################
    ##网络训练学习，对于每次学习
    ##############################
    # print 'len(test_data[0]) ',len(test_data[0])
    # print 'len(train_data): ',len(train_data[0])
    if(len(train_data[0])>12):
        # print '************************'
        CYCLE=100 #训练次数
    else:
        CYCLE = 150  # 训练次数
    input_gate=[0 for t in range(cell_num)]
    output_gate=[0 for t in range(cell_num)]
    forget_gate = [0 for t in range(cell_num)]
    forget_gate_input = [0 for t in range(cell_num)]
    Error_Cost = [0 for x in range(CYCLE)]  #误差消耗
    for iter in range(CYCLE):
        yita=0.1 #每次迭代权重调整比例
        for m in range(data_num):  #共四组训练值，第一个训练值不用加前边的细胞状态，以后的需要加前边的状态。
            cur_train_data = matiix_get_col(train_data, m)
            if(m==0): #前馈部分
                gate=matrix_tanhSingA(matrixMulSingleA(cur_train_data,weight_input_x))
                input_gate_input=matrixPlusSingle(matrixMulSingleA(cur_train_data,weight_inputgate_x),bias_input_gate)
                output_gate_input=matrixPlusSingle(matrixMulSingleA(cur_train_data,weight_outputgate_x),bias_output_gate)
                for n in range(cell_num):
                    input_gate[n]=1/(1+math.exp(-input_gate_input[n]))
                    output_gate[n]=1/(1+math.exp(-output_gate_input[n]))
                cur_cell_state=matrix_dot_mulSingle(input_gate,gate)
                mrtiix_col_alter(cell_state,m,cur_cell_state)
            else:
                befor_h_state=h_state[m-1]
                before_cell_state=matiix_get_col(cell_state, m-1)
                gate=matrix_tanhSingA(matrixPlusSingle(matrixMulSingleA(cur_train_data,weight_input_x),matrix_mul_numSingleA(weight_input_h,befor_h_state)))
                input_gate_input=matrixPlusSingle(matrixPlusSingle(matrixMulSingleA(cur_train_data,weight_inputgate_x),bias_input_gate),matrixMulSingleA(before_cell_state,weight_inputgate_c))
                forget_gate_input=matrixPlusSingle(matrixPlusSingle(matrixMulSingleA(cur_train_data,weight_forgetgate_x),bias_forget_gate),matrixMulSingleA(before_cell_state,weight_forgetgate_c))
                output_gate_input=matrixPlusSingle(matrixPlusSingle(matrixMulSingleA(cur_train_data,weight_outputgate_x),bias_output_gate),matrixMulSingleA(before_cell_state,weight_outputgate_c))

                for n in range(cell_num):
                    input_gate[n]=1/(1+math.exp(-input_gate_input[n]))
                    forget_gate[n] = 1 / (1 + math.exp(-forget_gate_input[n]))
                    output_gate[n] = 1 / (1 + math.exp(-output_gate_input[n]))
                cur_cell_state =matrixPlusSingle( matrix_dot_mulSingle(input_gate, gate),matrix_dot_mulSingle(before_cell_state,forget_gate))
                mrtiix_col_alter(cell_state, m, cur_cell_state)
            pre_h_state=matrix_tanhSingA(matrix_dot_mulSingle(cur_cell_state,output_gate))
            h_state[m]=matrixMulSingleAB(pre_h_state,weight_preh_h)  #计算此次预测值

            #误差计算
            # print 'h_state[m]: ',h_state[m]
            Error = h_state[m]-test_data[0][m]
            # Error_Cost[iter]+=Error**2
            Error_Cost[iter]+=abs(Error)**3
            #
            # if (Error < cost_gate):
            #     print '###############################'
            #     flag = 1
            #     break
            # else:
            weight_input_x, weight_input_h,weight_inputgate_x, weight_inputgate_c,weight_forgetgate_x, weight_forgetgate_c,weight_outputgate_x,weight_outputgate_c,weight_preh_h=LSTM_updata_weight(input_num,cell_num,output_num,m, yita, Error,weight_input_x,weight_input_h,weight_inputgate_x,weight_inputgate_c,weight_forgetgate_x,weight_forgetgate_c,weight_outputgate_x,weight_outputgate_c,weight_preh_h,cell_state, h_state,input_gate, forget_gate,output_gate, gate,train_data, pre_h_state,input_gate_input,output_gate_input,forget_gate_input)
        # print Error_Cost

        if (Error_Cost[iter] < cost_gate):
            print '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
            break
        # 没问题
    # print "Error_Cost", Error_Cost
        ## 预测第8周数据
        ## 数据加载
    temp=[0 for i in range(case.predict_data_range_days)]
    step=len(case.his_data_all[flavor_num])/case.predict_data_range_days
    yushu=len(case.his_data_all[flavor_num])%case.predict_data_range_days
    for x in range(len(case.his_data_all[flavor_num])-yushu):
            temp[x/step]+=float(case.his_data_all[flavor_num][x])
    mimi,maxi=Tools.getm(temp)
    for x in range(len(temp)):
        if (maxi - mimi <= 0.1):
            temp[x] = 0.0
        else:
            temp[x] = float((temp[x] - mimi) / (maxi - mimi))
    test_final = temp[len(temp)-input_num:len(temp)]
    # print 'test_final: ',test_final
    # 前馈
    m = data_num

    befor_h_state = h_state[m - 1]
    # print 'm:  ',m
    # 一个输出
    before_cell_state = matiix_get_col(cell_state, m - 1)

    gate = matrix_tanhSingA(matrixPlusSingle(matrixMulSingleA(test_final, weight_input_x),
                                             matrix_mul_numSingleA(weight_input_h, befor_h_state)))
    input_gate_input = matrixPlusSingle(
        matrixPlusSingle(matrixMulSingleA(test_final, weight_inputgate_x), bias_input_gate),
        matrixMulSingleA(before_cell_state, weight_inputgate_c))
    forget_gate_input = matrixPlusSingle(
        matrixPlusSingle(matrixMulSingleA(test_final, weight_forgetgate_x), bias_forget_gate),
        matrixMulSingleA(before_cell_state, weight_forgetgate_c))
    output_gate_input = matrixPlusSingle(
        matrixPlusSingle(matrixMulSingleA(test_final, weight_outputgate_x), bias_output_gate),
        matrixMulSingleA(before_cell_state, weight_outputgate_c))

    for n in range(cell_num):
        input_gate[n] = 1 / (1 + math.exp(-input_gate_input[n]))
        forget_gate[n] = 1 / (1 + math.exp(-forget_gate_input[n]))
        output_gate[n] = 1 / (1 + math.exp(-output_gate_input[n]))
    cur_cell_state = matrixPlusSingle(matrix_dot_mulSingle(input_gate, gate),
                                      matrix_dot_mulSingle(before_cell_state, forget_gate))
    pre_h_state = matrix_tanhSingA(matrix_dot_mulSingle(cur_cell_state, output_gate))
    h_state_final = matrixMulSingleAB(pre_h_state, weight_preh_h)  # 计算此次预测值
    # print '归一化h_state_final:',h_state_final
    h_state_final = h_state_final * (maxi-mimi) + mimi
    if(h_state_final<=0.0):
        h_state_final=0.0


    h_state_final = round(h_state_final)
    # print "h_state_final",h_state_final
    # print iter

    return h_state_final

