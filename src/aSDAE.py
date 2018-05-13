import tensorflow as tf
import numpy as np
import pandas as pd

c = np.array([[1,-1,2],[-4,6,-5]],dtype=np.float32)
C = tf.Variable(c)
clip_C = C.assign(tf.maximum(tf.zeros_like(C), C))
# clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))


# print(C)

np.random.seed(0)
A_orig = np.array([[7,2,1,4],[2,4,8,6],[4,3,2,7]],dtype=np.float32)

A_orig_df = pd.DataFrame(A_orig)
A_df_masked = A_orig_df.copy()
print(A_df_masked)
A_df_masked.iloc[0,0]=np.NAN
print(A_df_masked)
np_mask = A_df_masked.notnull()
print(np_mask.values)

# Boolean mask for computing cost only on valid (not missing) entries
tf_mask = tf.Variable(np_mask.values)


A = tf.constant(A_df_masked.values)
print(A_df_masked.values)
shape = A_df_masked.values.shape

#latent factors
rank = 3 

# Initializing random H and W
temp_H = np.random.randn(rank, shape[1]).astype(np.float32)
# temp_H = np.divide(temp_H, temp_H.max())

temp_W = np.random.randn(shape[0], rank).astype(np.float32)
# temp_W = np.divide(temp_W, temp_W.max())

H =  tf.Variable(temp_H)
W = tf.Variable(temp_W)
WH = tf.matmul(W, H)

print(tf.boolean_mask(A, tf_mask))

# cost = tf.reduce_sum(tf.pow(tf.boolean_mask(A, tf_mask) - tf.boolean_mask(WH, tf_mask), 2))
cost = tf.reduce_sum(tf.square(tf.subtract(WH,A)))
# Learning rate
lr = 0.001
# Number of steps
steps = 1000
train_step = tf.train.AdamOptimizer(lr).minimize(cost)
init = tf.global_variables_initializer()


# Clipping operation. This ensure that W and H learnt are non-negative
clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
clip = tf.group(clip_W, clip_H)

steps = 10000
with tf.Session() as sess:
    sess.run(init)
#     print(sess.run(C))
#     print(sess.run(clip_C))
    print(sess.run(A))
    print(sess.run(tf_mask))
    print(sess.run(tf.boolean_mask(A, tf_mask)))
    print(sess.run(tf.reduce_sum(tf.square(tf.subtract(A, WH)))))
    print(sess.run(tf.reduce_sum(tf.pow(tf.boolean_mask(A, tf_mask) - tf.boolean_mask(WH, tf_mask), 2))))
    for i in range(steps):
        sess.run(train_step)
        sess.run(clip)
        if i%1000==0:
            print("\nCost: %f" % sess.run(cost))
            print("*"*40)
    learnt_W = sess.run(W)
    learnt_H = sess.run(H)
pred = np.dot(learnt_W, learnt_H)
pred_df = pd.DataFrame(pred)
print(pred_df)
print("*"*40)
print(A_orig_df)
