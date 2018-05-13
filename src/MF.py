import tensorflow as tf
import numpy as np

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    high = constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform([fan_in,fan_out], minval=low, maxval=high, dtype=tf.float32)

class DenoisingAutoEncoder(object):
    def __init__(self, lr = 0.01, alpha = 0.1, activation="sigmoid", optimizer="adam", loss="mse", scale = 0.1):
        self.lr = lr
        self.alpha = alpha
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.scale = scale
        
    def train(self, user_item_ratings_clean, user_info_clean, item_info_clean, dims=[8,6,4,6,8]):
        user_ratings_clean = np.copy(user_item_ratings_clean)
        user_ratings_noisy = self.add_noise(user_ratings_clean)
        user_info_noisy = self.add_noise(user_info_clean)
        
        item_ratings_clean = np.transpose(user_item_ratings_clean)
        item_ratings_noisy = self.add_noise(item_ratings_clean)
        item_info_noisy = self.add_noise(item_info_clean)
        
        c_v = self.add_noise(np.transpose(np.copy(r)))
        c_s_u = self.add_noise(np.copy(s_u))
        c_s_v = self.add_noise(np.copy(s_v))
        
        u_weights_bias = []
        v_weights_bias = []
        u_f = self.init_weights_bias(len(c_r[0]),len(c_s_u[0]),dims[0])
        v_f = self.init_weights_bias(len(c_v[0]),len(c_s_v[0]),dims[0])
        u_weights_bias.append(u_f)
        v_weights_bias.append(v_f)
        for i in range(len(dims)-1):
            u_f = self.init_weights_bias(dims[i],len(c_s_u[0]),dims[i+1])
            v_f = self.init_weights_bias(dims[i],len(c_s_v[0]),dims[i+1])
            u_weights_bias.append(u_f)
            v_weights_bias.append(v_f)
        u_f = self.init_weights_bias(len(c_r[0]),len(c_s_u[0]),dims[-1],True) 
        v_f = self.init_weights_bias(len(c_v[0]),len(c_s_v[0]),dims[-1],True)    
        u_weights_bias.append(u_f)
        v_weights_bias.append(v_f)
        
        p_c_r = tf.placeholder(dtype=tf.float32, shape=[None, len(c_r[0])], name="p_c_r")
        p_s_u = tf.placeholder(dtype=tf.float32, shape=[None, len(c_s_u[0])], name="p_s_u")
        p_c_v = tf.placeholder(dtype=tf.float32, shape=[None, len(c_v[0])], name="p_c_v")
        p_s_v = tf.placeholder(dtype=tf.float32, shape=[None, len(c_s_v[0])], name="p_s_v")
        
        U = self.encode(p_c_r, p_s_u, u_weights_bias)
        V = self.encode(p_c_v, p_s_v, v_weights_bias)
        
        decode_r, decode_s_u = self.decode(U,u_weights_bias)
        decode_v, decode_s_v = self.decode(V,v_weights_bias)
        
        loss_mf = tf.reduce_sum(tf.square(tf.subtract(r, tf.matmul(U, tf.transpose(V)))))
        loss_u = tf.reduce_sum(tf.square(tf.subtract(r, decode_r)))+tf.reduce_sum(tf.square(tf.subtract(s_u, decode_s_u)))
        loss_v = tf.reduce_sum(tf.square(tf.subtract(c_v, decode_v)))+tf.reduce_sum(tf.square(tf.subtract(s_v, decode_s_v)))
        loss = loss_mf+loss_u+loss_v
#         loss = loss_u
        train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(10000):
                feed_data = {p_c_r:r,p_s_u:s_u,p_c_v:c_v,p_s_v:s_v}
                _,cost,u,v= sess.run((train_op,loss,U,V),feed_dict=feed_data)
                if (epoch+1)% 1000 == 0:
                    print("\nCost: %f" % cost)
                    print("*"*40)
            return u,v
        
    def encode(self, u, s_u, u_weights_bias):
        encode_u = tf.matmul(u, u_weights_bias[0]["coder_w"])
        encode_s_u = tf.matmul(s_u, u_weights_bias[0]["s_coder_w"])
        u_tensor = encode_u + encode_s_u + u_weights_bias[0]["coder_b"]
        encode_u_r = self.activate(u_tensor)
        
        for i in range(1,len(u_weights_bias)-1):
            encode_u = tf.matmul(encode_u_r, u_weights_bias[i]["coder_w"])
            encode_s_u = tf.matmul(s_u, u_weights_bias[i]["s_coder_w"])
            u_tensor = encode_u + encode_s_u + u_weights_bias[i]["coder_b"]
            encode_u_r = self.activate(u_tensor)
        return encode_u_r
    
    def decode(self, u, u_weights_bias):
        decode_u = tf.add(tf.matmul(u, u_weights_bias[-1]["coder_w"]), u_weights_bias[-1]["coder_b"])
        decode_s_u = tf.add(tf.matmul(u, u_weights_bias[-1]["s_coder_w"]), u_weights_bias[-1]["s_coder_b"])
#         decode_u_r = self.activate(decode_u)
#         decode_s_u_r = self.activate(decode_s_u)
        return decode_u, decode_s_u
        
    def activate(self, x, name=None):
        #激活神经元
        if self.activation == "sigmoid":
            return tf.nn.sigmoid(x, name)
        if self.activation == "tanh":
            return tf.nn.tanh(x, name)
        if self.activation == "relu":
            return tf.nn.relu(x, name)
        else:
            print("Invalid activation!")
            return x
            
    def init_weights_bias(self, n_input, s_input, n_output, isLast = False):
        #权重与偏置初始化
        weights_bias  = dict()
        if isLast:
            weights_bias['coder_w'] = tf.Variable(xavier_init(n_output,n_input))
            weights_bias['s_coder_w'] = tf.Variable(xavier_init(n_output,s_input))
            weights_bias['coder_b'] = tf.Variable(tf.truncated_normal([n_input],dtype=tf.float32))
            weights_bias['s_coder_b'] = tf.Variable(tf.truncated_normal([s_input],dtype=tf.float32))
        else:
            weights_bias['coder_w'] = tf.Variable(xavier_init(n_input,n_output))
            weights_bias['s_coder_w'] = tf.Variable(xavier_init(s_input,n_output))
            weights_bias['coder_b'] = tf.Variable(tf.truncated_normal([n_output],dtype=tf.float32))
        return weights_bias    
        
    def add_noise(self, x_):
        #添加高斯噪声
        n = np.random.normal(0, self.scale, (len(x_), len(x_[0])))
        return x_ + n.astype(np.float32)
        
