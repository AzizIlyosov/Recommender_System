import numpy as np
import pymysql
import pandas as pd
import tensorflow as tf


def SGDTrain(dataMat, K):
    maxCycles = 10000
    alpha = 0.002
    beta = 0.02
    m,n = dataMat.shape;
    print(m,n)
    
    P = np.random.random((m,K))
    Q = np.random.random((K,n))
    
    for count in range(maxCycles):
        loss = 0.0
        for i in range(m):
            for j in range(n):
                if dataMat[i,j] > 0:
                    err = dataMat[i,j]
                    for k in range(K):
                        err -= P[i,k]*Q[k,j]
                    regularizer = 0.0
                    predict = 0.0
                    for k in range(K):
                        t1 = P[i,k]
                        t2 = Q[k,j]
                        P[i,k] = t1+alpha*(2*err*t2 - beta*t1)
                        Q[k,j] = t2+alpha*(2*err*t1 - beta*t2)
                        predict += P[i,k]*Q[k,j]
                        regularizer += P[i,k]*P[i,k] + Q[k,j]*Q[k,j]
                    loss = (dataMat[i,j] - predict)* (dataMat[i,j] - predict)+ (beta * regularizer / 2)
        if loss < 0.001:
            break
        if count % 1000 == 0:
            print(loss)
            
    return P,Q
def getLoss(m, n):
#     m是原始矩阵

    return


def train(x, k):
    lr = 0.001
    o = tf.placeholder(dtype=tf.float32, shape=[3, 4])
    w = np.random.randn(3, k).astype(np.float32)
    h = np.random.randn(k, 4).astype(np.float32)
    W =  tf.Variable(w)
    H =  tf.Variable(h)
#     ost = tf.reduce_sum(tf.pow(tf.boolean_mask(A, tf_mask) - tf.boolean_mask(WH, tf_mask), 2))
    loss = tf.reduce_mean(tf.square(tf.subtract(tf.matmul(W,H), o)))
#     loss = tf.reduce_sum(tf.pow(o - tf.matmul(W,H), 2))
    clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
    clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
    clip = tf.group(clip_W, clip_H)
    
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5000):
            _,_, cost = sess.run((train_op,clip,loss), feed_dict={o:x})
            if (i+1) % 1000 == 0:
                print(cost)
                z=sess.run(W) 
                v=sess.run(H)
#         print(z)
#         print(v)
        print(x)
        print(np.dot(z,v))
#     
    return     
    
if __name__ == '__main__':
#     V = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    arr = [[7,0,1,4],[2,4,8,6],[4,3,2,7]]
    df = pd.DataFrame(arr)
    c = df.replace(0, np.NAN)
    print(c)
#     dataMat = np.mat(arr)
#     train(dataMat, 3)
#     print(np.shape(dataMat)[0])
#     P, Q = SGDTrain(dataMat,2)
#     print(P)
#     print(Q)
#     print(np.dot(P,Q))
    