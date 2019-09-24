# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 20:30:21 2019

@author: x00423910
"""


import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random
from matplotlib import pyplot as plt

LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 1.0 # 0.99
REGULARIZER = 0.001
MOVING_AVERAGE_DECAY = 0.8 # 0.99

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def plot_history(epoch,history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Root Mean Square Error')
    plt.plot(epoch, np.array(history),
                label='Train_Loss')

    plt.legend()
    plt.ylim([0, 100])
    plt.savefig('loss.png')

def preProcessDataForTraining():
    #导入训练数据集
    total_train_set_data = []
    path_="pin"
    for train_set_file_name in os.listdir(path_):
        with open(os.path.join(path_,train_set_file_name), "r") as rf:
            file_data = pd.read_csv(rf)
            file_data = np.array(file_data.get_values(), dtype=np.float32)
            print("fileName:", train_set_file_name, "  shape of file data:", file_data.shape)
            total_train_set_data.extend(file_data)
    total_train_set_data = np.array(total_train_set_data)
    print("shape of total_train_set_data data:", total_train_set_data.shape)
    #至此，我们把“train_set”文件夹下的数据都读到total_train_set_data中了
    #此时的total_train_set_data是个维度为[15284, 18]的数组
    #开始对每个数据样本构造数据特征
    total_train_xs = []
    total_train_ys = []
    for i in range(len(total_train_set_data)):
        org_data = total_train_set_data[i]
        #假设我们觉得数据中的四个地理坐标可以构造一个距离特征
        feature_dis_2d = np.sqrt(np.power(org_data[12]-org_data[1],2)+np.power(org_data[13]-org_data[2],2))     
        #假设我们觉得RS Power这个特征也很重要
        feature_RS_Power = org_data[8]
        #假设我们觉得三维空间中的这个距离也很重要
        # feature_dis_3d = np.sqrt(np.power(org_data[12]-org_data[1],2)
        #                          +np.power(org_data[13]-org_data[2],2)
        #                          +np.power(org_data[14]-org_data[9],2))
        feature_height=org_data[3]-feature_dis_2d*np.tan((org_data[5]+org_data[6])*np.pi/180)+(org_data[9]-org_data[14])
        feature_frequence=org_data[7]
        feature_clutter_index=org_data[16]
        feature_Azimuth=org_data[4]#水平角度
        feature_cell_clutter_index=org_data[11]
        feature_cell_build=org_data[10]
        feature_build=org_data[15]

        tmp_train_xs = [feature_dis_2d/100, feature_RS_Power,feature_height/100,feature_frequence,feature_clutter_index,
                        feature_Azimuth,feature_cell_clutter_index,feature_cell_build,feature_build]
        #我们期望预测的就是第18列的RSRP

        tmp_train_ys = [org_data[17]]
        total_train_xs.append(tmp_train_xs)
        total_train_ys.append(tmp_train_ys)
    #将total_train_xs、total_train_ys转换为numpy数组类型

    total_train_xs = np.array(total_train_xs)
    # total_train_xs=normalization(total_train_xs)
    # mu = np.mean(total_train_xs, axis=0)
    # sigma = np.std(total_train_xs, axis=0)
    # total_train_xs= (total_train_xs - mu) / sigma


    total_train_ys = np.array(total_train_ys)    
    print("shape of total_train_xs:", total_train_xs.shape)    
    print("shape of total_train_ys:", total_train_ys.shape)
    
    return total_train_xs, total_train_ys


def weight_variable(shape, regularizer):
	w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w


def bias_variable(shape):
    # 本例中用relu激活函数，所以用一个很小的正偏置较好
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

px=[]
py=[]
def train():
    #******* 1. 构建网络 *******    
    INPUT_FEATURE_NUM = 9  #因为预处理中只构造了9个特征
    OUTPUT_FEATURE_NUM = 1 #因为要预测的只有1个值
    x = tf.placeholder(tf.float32, [None, INPUT_FEATURE_NUM], name = "haha_input_x")
    #构造输错节点的Placeholder,用来接收标准答案数据的
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_FEATURE_NUM])
    #最简单的网络，就是一个矩阵乘法 y = x*W+b，
    #其中X是我们的Placeholder，用来接输入数据的
    #y是整个网络的输出节点
    # W = tf.Variable(tf.random_normal([INPUT_FEATURE_NUM, OUTPUT_FEATURE_NUM]))
    # b = tf.Variable(tf.random_normal([OUTPUT_FEATURE_NUM]))
    # y = tf.add( tf.matmul(x, W), b, name = "haha_output_y")

    # FC1
    W_fc1 = weight_variable([INPUT_FEATURE_NUM, 64], REGULARIZER)
    b_fc1 = bias_variable([64])
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)


    W_fc2 = weight_variable([64, 64], REGULARIZER)
    b_fc2 = bias_variable([64])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    #
    #
    W_fc3 = weight_variable([64, 64], REGULARIZER)
    b_fc3 = bias_variable([64])
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)


    W_fcn = weight_variable([64, OUTPUT_FEATURE_NUM], REGULARIZER)
    b_fcn = bias_variable([OUTPUT_FEATURE_NUM])
    y = tf.add(tf.matmul(h_fc3, W_fcn) , b_fcn,name = "haha_output_y")

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        1000,
        LEARNING_RATE_DECAY,
        staircase=True)


    # # 1.损失函数：cross_entropy
    cost = tf.reduce_mean(tf.sqrt(tf.pow(y - y_, 2))) # + tf.add_n(tf.get_collection('losses'))
    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y_pre))
    # 2.优化函数：AdamOptimizer, 优化速度要比 GradientOptimizer 快很多
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())

    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
    

    # #构造训练网络的cost,就是看预期答案y_和实际推理出来的答案y的差距是多少
    # cost = tf.reduce_mean(tf.sqrt(tf.pow(y-y_,2)))
    # #选择优化的算法使最小化cost
    # train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cost)
        

    
    #开始训练
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        #导入预处理之后的训练数据
        total_train_xs, total_train_ys = preProcessDataForTraining()
        for i in range(12000):
            #在total_train_xs, total_train_ys数据集中随机抽取batch_size个样本出来
            #作为本轮迭代的训练数据batch_xs, batch_ys
            batch_size = 64
            sample_idxs = random.choices(range(len(total_train_xs)), k=batch_size)
            batch_xs = []
            batch_ys = []
            for idx in sample_idxs:
                batch_xs.append(total_train_xs[idx])
                batch_ys.append(total_train_ys[idx])
            batch_xs = np.array(batch_xs)
            batch_ys = np.array(batch_ys)            
            #喂训练数据进去训练
            #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            #看经过这次训练后，cost的值是多少
            _, cost_value, step= sess.run([train_op, cost, global_step], feed_dict={x: batch_xs, y_: batch_ys})
            print("after iter:",i, " cost:", cost_value)
            if i>= 6000:
                if cost_value<4.7:
                    break
            px.append(i)
            py.append(cost_value)

            # plot_history(i,cost_value)
        #训练完成之后，调用函数将网络保存成tensorflow的SavedModel格式的模型
        #注意这里保存的模型签名，就是告诉别人我们模型的输入和输出
        #如下就是告诉别人我们这个模型只有一个输入，这个输入就是x节点，别人用这个模型的
        #时候就要准备一个输入的Map，这个Map里面有一个key为“myInput”的项，对应的值应该
        #是一个维度为[None, INPUT_FEATURE_NUM]的numpy数组;
        #同理，输出的签名就是在告诉别人调用我们这个模型后返回的数据是一个Map，其中有一
        #key为"myOutput"的项，该项的值是一个维度为[None, OUTPUT_FEATURE_NUM]的numpy数
        #组;
        #这个签名是通过x和y节点的来告诉别人这个模型的输入输出的数据格式和维度
        #其中x作为一个数据输入节点，必须是placeholder类型，从上面也可以看到其维度是我们
        #自己根据自己的数据特征设计的，而y的维度是由x维度是[None, INPUT_FEATURE_NUM]
        #又乘了一个[INPUT_FEATURE_NUM, OUTPUT_FEATURE_NUM]的W矩阵再加上一个[OUTPUT_FEATURE_NUM]
        #的b矩阵得出来的，所以y的维度是[None, OUTPUT_FEATURE_NUM];
        #注意上的y_也是一个维度是[None, OUTPUT_FEATURE_NUM]的placeholder，是为了用于喂y在训练集中的预期答案，
        #再跟预测出来的y进行计算得到优化目标cost函数的。
        #注意区分签名里的outputs里的是输出节点y,而不是用了喂预期数据的y_,因为在模型训练好，用于推理
        #的时候，你是无法得到测试集中的预期答案的。
        tf.saved_model.simple_save(
                sess,
                "./model",
                inputs={"myInput":x}, #这个就是我们的模型签名，告诉别人我们模型输入是x节点
                outputs={"myOutput":y} #这个就是我们的模型签名，告诉别人我们模型输入是y节点
                )
        
        
train()
plot_history(px,py)

