#coding:utf-8
import random
import math
# 参考博客 ： http://www.cnblogs.com/charlotte77/p/5629865.html
def hidden_layer(input_layer,hidden_layer_weights,hidden_layer_bias):
    '''
    正向-计算隐藏层
    input_layer 输入
    hidden_layer_weights 权重
    hidden_layer_bias 截距项
    '''
    hl = []
    hlws = []
    for i in range(0,len(hidden_layer_weights),len(input_layer)):
        hlws.append(hidden_layer_weights[i:i+len(input_layer)])  
 
    for hlw in hlws:
        hli = 0
        for i in range(len(hlw)):
            hli += input_layer[i]*hlw[i]
        hli += hidden_layer_bias * 1
        hli = 1/(1+math.exp(-hli))
        hl.append(hli)
    return hl


def square_error(outputs,targets):
    '''
    计算误差
    outputs 输出列表
    targets 目标列表
    '''
    e_total = 0
    for i in range(len(outputs)):
        e = (targets[i]-outputs[i])**2/2
        e_total += e
    return e_total
        
        
def derivative_out(weights,targets,outs,hls):
    '''
    计算输出层的权值更新值
    weights 权重列表
    ∂E/∂w = ∂E/∂out * ∂out/∂net * ∂net/∂w  
    ''' 
    hlws = []
    for i in range(0,len(weights),len(outs)):
        hlws.append(weights[i:i+len(outs)])


    wplus = []
    for i in range(len(hlws)):
        for j in range(len(hlws[i])):
            wplu = 0
            derivative_1 = -(targets[i] - outs[i])
            derivative_2 = (outs[i] * (1 - outs[i]))
            derivative_3 = hls[i]
            dt = derivative_1 * derivative_2 * derivative_3
            wplu = hlws[i][j] - dt * 0.5
            wplus.append(wplu)
    return wplus


def derivative_hidden(inps,weights,targets,outs,hous,out_weights):
    '''
    计算隐藏层的权值更新值
    weights 权重列表
    ∂E/∂w = ∂E/∂out * ∂out/∂net * ∂net/∂w  
    ''' 
    hlws = []
    for i in range(0,len(weights),len(outs)):
        hlws.append(weights[i:i+len(outs)])
    olws = []
    for i in range(0,len(out_weights),len(outs)):
        olws.append(out_weights[i:i+len(outs)])

    wplus = []
    for i in range(len(hlws)):
        for j in range(len(hous)):
            dt_1 = sum(e_derivative(targets,outs,olws[i]))
            dt_2 = (hous[j] * (1 - hous[j]))
            dt_3 = inps[i]
           
            dt = dt_1 * dt_2 * dt_3 
            wplu = hlws[i][j] - dt * 0.5
            wplus.append(wplu)    
    return wplus
        
       
    
def e_derivative(targets,outs,weights):
    dts = []
    for i in range(len(outs)):
        derivative_1 = -(targets[i] - outs[i])
        derivative_2 = (outs[i] * (1 - outs[i]))
        derivative_3 = weights[i]
        dt = derivative_1 * derivative_2 * derivative_3 
        dts.append(dt)
    return dts
        
def train(inputs,hw,hb,ow,ob,targets,count):
    '''
    训练权重
    '''
    for i in range(0,20000):    
        #计算隐藏层数据
        hout=hidden_layer(inputs,hw,hb) 
        #print(hl)
        #计算输出层数据
        out=hidden_layer(hout,ow,ob)    
        print(out)
        #计算误差
        se = square_error(out,targets)
        print('误差----',se)
        #反向传播 更新隐藏层到输出层权重
        dtos = derivative_out(ow,targets,out,hout)
        #print(dtos)
        #反向传播 更新输入层到隐藏层权重
        dths = derivative_hidden(inputs,hw,targets,out,hout,ow)
        #print(dths)
        #更新权重
        ow = dtos
        hw = dths  
    return {'hw':hw,'ow':ow}     

if __name__ == '__main__':
    #输入数据
    inputs = [0.05,0.10]
    #输入层到隐藏层 权重
    hw = [0.15,0.20,0.25,0.30]
    hb = 0.35
    #隐藏层到输出层 权重
    ow = [0.40,0.45,0.50,0.55]
    ob = 0.60
    targets = [0.01,0.99]
    train(inputs,hw,hb,ow,ob,targets,1000)
        


    