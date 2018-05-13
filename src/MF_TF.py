import numpy as np
import tensorflow as tf


def train(M,k,lr=0.001,epoches=5000):
    X = tf.constant(M)
    U = tf.Variable(np.random.randn(M.shape[0], k).astype(np.float32))
    V = tf.Variable(np.random.randn(k, M.shape[1]).astype(np.float32))
    
    clip_U = U.assign(tf.maximum(tf.zeros_like(U), U))
    clip_V = V.assign(tf.maximum(tf.zeros_like(V), V))
    clip = tf.group(clip_U, clip_V)
    
    loss = tf.reduce_sum(tf.square(tf.subtract(tf.matmul(U,V), X)))
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoches):
            _,_,cost = sess.run((train_op,clip,loss),feed_dict={X:M})
            if (epoch+1)% 1000 == 0:
                print("\nCost: %f" % cost)
                print("*"*40)
        return sess.run(U), sess.run(V)

if __name__ == '__main__':
    M = np.array([[7,2,1,4],[2,4,8,6],[4,3,2,7]],dtype=np.float32)
#     print(M.shape[0])
#     print(np.shape(M)[0])
    U,V = train(M, 3)
    print(M)
    print(np.dot(U,V))
    