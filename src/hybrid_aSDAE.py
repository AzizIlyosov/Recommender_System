import tensorflow as tf
import numpy as np

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    high = constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform([fan_in,fan_out], minval=low, maxval=high, dtype=tf.float32)
#     return tf.random_normal([fan_in,fan_out],dtype=tf.float32)

class DenoisingAutoEncoder(object):
    def __init__(self, lr = 0.004, rating_loss_weight = 0.8, l2_loss_weight=0.0, activation="sigmoid", optimizer="adam", loss="mse", scale = 0.1):
        self.lr = lr
        self.rating_loss_weight = rating_loss_weight
        self.l2_loss_weight = l2_loss_weight
        self.l2_reg = tf.nn.l2_loss
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.scale = scale
        
    def train(self, user_item_ratings_clean, user_info_clean, item_info_clean, dims=[8,6,4,6,8]):
        user_ratings_clean = np.copy(user_item_ratings_clean)
        user_ratings_noisy = self.add_noise(user_ratings_clean)
        user_ratings_dims = len(user_ratings_noisy[0])
        user_ratings_mask = user_ratings_clean > 0
        user_info_noisy = self.add_noise(user_info_clean)
        user_info_dims = len(user_info_noisy[0])
        
        item_ratings_clean = user_item_ratings_clean.T
        item_ratings_noisy = self.add_noise(item_ratings_clean)
        item_ratings_dims = len(item_ratings_noisy[0])
        item_ratings_mask = user_ratings_mask.T
        item_info_noisy = self.add_noise(item_info_clean)
        item_info_dims = len(item_info_noisy[0])
         
        user_weights_bias = []
        item_weights_bias = []
        
        u_w_b = self.init_weights_bias(user_ratings_dims, user_info_dims, dims[0], 1, False, "user")
        i_w_b = self.init_weights_bias(item_ratings_dims, item_info_dims, dims[0], 1, False, "item")
        user_weights_bias.append(u_w_b)
        item_weights_bias.append(i_w_b)
        
        for i in range(len(dims)-1):
            u_w_b = self.init_weights_bias(dims[i], user_info_dims, dims[i+1], (i+2), False, "user")
            i_w_b = self.init_weights_bias(dims[i], item_info_dims, dims[i+1], (i+2), False, "item")
            user_weights_bias.append(u_w_b)
            item_weights_bias.append(i_w_b)
        u_w_b = self.init_weights_bias(user_ratings_dims, user_info_dims, dims[-1], (len(dims)+1), True, "user")
        i_w_b = self.init_weights_bias(item_ratings_dims, item_info_dims, dims[-1], (len(dims)+1), True, "item")
        user_weights_bias.append(u_w_b)
        item_weights_bias.append(i_w_b)
        
        p_user_ratings_clean = tf.placeholder(dtype=tf.float32, shape=[None, user_ratings_dims],
                                              name='p_user_ratings_clean')
        p_user_ratings_noisy = tf.placeholder(dtype=tf.float32, shape=[None, user_ratings_dims],
                                              name='p_user_ratings_noisy')
        p_user_ratings_mask = tf.placeholder(dtype=tf.float32, shape=[None, user_ratings_dims],
                                              name='p_user_ratings_mask')
        p_user_info_clean = tf.placeholder(dtype=tf.float32, shape=[None, user_info_dims],
                                              name='p_user_info_clean')
        p_user_info_noisy = tf.placeholder(dtype=tf.float32, shape=[None, user_info_dims],
                                              name='p_user_info_noisy')
        
        p_item_ratings_clean = tf.placeholder(dtype=tf.float32, shape=[None, item_ratings_dims],
                                              name='p_item_ratings_clean')
        p_item_ratings_noisy = tf.placeholder(dtype=tf.float32, shape=[None, item_ratings_dims],
                                              name='p_item_ratings_noisy')
        p_item_ratings_mask = tf.placeholder(dtype=tf.float32, shape=[None, item_ratings_dims],
                                              name='p_item_ratings_mask')
        p_item_info_clean = tf.placeholder(dtype=tf.float32, shape=[None, item_info_dims],
                                              name='p_item_info_clean')
        p_item_info_noisy = tf.placeholder(dtype=tf.float32, shape=[None, item_info_dims],
                                              name='p_item_info_noisy')
        
        user_bottleneck_op = self.encode(p_user_ratings_noisy, p_user_info_noisy, user_weights_bias,"user")
        item_bottleneck_op = self.encode(p_item_ratings_noisy, p_item_info_noisy, item_weights_bias,"item")
        
        user_ratings_pred, user_info_pred = self.decode(user_bottleneck_op, p_user_info_noisy, user_weights_bias,
                                                    "user")
        item_ratings_pred, item_info_pred = self.decode(item_bottleneck_op, p_item_info_noisy, item_weights_bias,
                                                    "item")
        
        def reconstruction_loss(ratings_clean, ratings_pred, ratings_mask, info_clean, info_pred, name):
            with tf.variable_scope("reconstruction_loss" + name):
#                 loss_rating = tf.losses.mean_squared_error(
#                     labels=ratings_clean,
#                     predictions=ratings_pred)
#                 loss_rating = tf.losses.mean_squared_error(
#                     labels=tf.boolean_mask(ratings_clean, ratings_mask, name='label_mask_' + name),
#                     predictions=tf.boolean_mask(ratings_pred, ratings_mask, name='prediction_mask_' + name))
                loss_rating = tf.losses.mean_squared_error(
                    labels=tf.multiply(ratings_clean, ratings_mask, name='label_mask_' + name),
                    predictions=tf.multiply(ratings_pred, ratings_mask, name='prediction_mask_' + name))
                loss_info = tf.losses.mean_squared_error(labels=info_clean, predictions=info_pred)
                reconstruct_loss = self.rating_loss_weight * loss_rating + (1-self.rating_loss_weight)*loss_info
            return reconstruct_loss

        def get_l2_loss(): 
#             l2_loss = 0
            l2_loss = tf.losses.get_regularization_loss() * self.l2_loss_weight
#             l2_loss *= self.l2_loss_weight
#             l2_loss = tf.add_n(tf.get_collection("losses"))      
            return l2_loss
        
        def get_rating_loss(ratings, ratings_mask, user_bottleneck, item_bottleneck, name):
            with tf.variable_scope("rating_loss_" + name):
                rating_pred = tf.matmul(user_bottleneck, tf.transpose(item_bottleneck), name="rating_pred")
#                 rating_loss = tf.losses.mean_squared_error(labels=ratings,
#                         predictions=rating_pred)
#                 c = tf.boolean_mask(ratings, ratings_mask)
#                 d = tf.boolean_mask(rating_pred, ratings_mask)
                rating_loss = tf.losses.mean_squared_error(labels=ratings,
                        predictions=tf.multiply(rating_pred, ratings_mask, name='pred_mask_' + name))
            return rating_loss
        
        def total_loss(user_ratings_clean, user_ratings_pred, user_ratings_mask, user_info_clean, user_info_pred, user_bottleneck, 
                       item_ratings_clean, item_ratings_pred, item_ratings_mask, item_info_clean, item_info_pred, item_bottleneck,
                       name):
            with tf.variable_scope("total_loss_" + name): 
                reconstruct_loss_user = reconstruction_loss(user_ratings_clean, user_ratings_pred, user_ratings_mask, user_info_clean, user_info_pred, name="user")    
                reconstruct_loss_item = reconstruction_loss(item_ratings_clean, item_ratings_pred, item_ratings_mask, item_info_clean, item_info_pred, name="item")
                l2_loss = get_l2_loss()
                rating_loss = get_rating_loss(user_ratings_clean, user_ratings_mask, user_bottleneck, item_bottleneck, name)
                loss = reconstruct_loss_user + reconstruct_loss_item + rating_loss + l2_loss
            return loss
        
        loss = total_loss(p_user_ratings_clean, user_ratings_pred, p_user_ratings_mask, p_user_info_clean, user_info_pred, user_bottleneck_op, 
                          p_item_ratings_clean, item_ratings_pred, p_item_ratings_mask, p_item_info_clean, item_info_pred, item_bottleneck_op, name="all")
        
        train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(20000):
                feed_data = {p_user_ratings_clean: user_ratings_clean,
                         p_user_ratings_noisy: user_ratings_noisy,
                         p_user_ratings_mask: user_ratings_mask,
                         p_user_info_clean: user_info_clean,
                         p_user_info_noisy: user_info_noisy,
                         p_item_ratings_clean: item_ratings_clean,
                         p_item_ratings_noisy: item_ratings_noisy,
                         p_item_ratings_mask: item_ratings_mask,
                         p_item_info_clean: item_info_clean,
                         p_item_info_noisy: item_info_noisy,
                         }
                _,cost,u,v,pred_u= sess.run((train_op,loss,user_bottleneck_op,item_bottleneck_op, user_ratings_pred),feed_dict=feed_data)
                if (epoch+1)% 5000 == 0:
                    print("\nCost: %f" % cost)
                    print("*"*40)
            return u,v,pred_u
        
    def encode(self, user_ratings, user_info, user_weights_bias, name):
        with tf.variable_scope(name + '_encoder'):
            mid = int(len(user_weights_bias)/2)
            encode_u = tf.matmul(user_ratings,user_weights_bias[0]["coder_w"])
            encode_u_i = tf.matmul(user_info,user_weights_bias[0]["s_coder_w"])
            u_tensor = encode_u + encode_u_i + user_weights_bias[0]["coder_b"]
            encode_result = self.activate(u_tensor)
            for i in range(1,mid):
                encode_u = tf.matmul(encode_result,user_weights_bias[i]["coder_w"])
                encode_u_i = tf.matmul(user_info,user_weights_bias[i]["s_coder_w"])
                u_tensor = encode_u + encode_u_i + user_weights_bias[i]["coder_b"]
                encode_result = self.activate(u_tensor)
                
        return encode_result
    
    def decode(self, user_bottleneck_op, user_info, user_weights_bias, name):
        with tf.variable_scope(name + '_decoder'):
            mid = int(len(user_weights_bias)/2)
            decode_result = user_bottleneck_op
            for i in range(mid, len(user_weights_bias)-1):
#                 print(user_weights_bias[i]["coder_w"])
#                 encode_u = tf.matmul(decode_result,user_weights_bias[i]["coder_w"])
#                 encode_u_i = tf.matmul(user_info,user_weights_bias[i]["s_coder_w"])
#                 u_tensor = encode_u + encode_u_i + user_weights_bias[i]["coder_b"]
#                 decode_result = self.activate(u_tensor)
                encode_weights_bias = []
                encode_weights_bias.append(user_weights_bias[i])
                decode_result = self.encode(decode_result, user_info, encode_weights_bias, name+"_decoder")
            user_rating_tensor = tf.add(tf.matmul(decode_result, user_weights_bias[-1]["coder_w"]), user_weights_bias[-1]["coder_b"])
            user_infor_tensor = tf.add(tf.matmul(decode_result, user_weights_bias[-1]["s_coder_w"]), user_weights_bias[-1]["s_coder_b"])
#             user_rating_pred = self.activate(user_rating_tensor)
#             user_info_pred = self.activate(user_infor_tensor)
            user_rating_pred = user_rating_tensor
            user_info_pred = user_infor_tensor
        return user_rating_pred, user_info_pred
        
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
            
    def init_weights_bias(self, n_input, s_input, n_output, n_layer, isLast = False, name="user"):
        #权重与偏置初始化
        pre_name = name + '_'+str(n_layer)
        with tf.variable_scope(pre_name):
            weights_bias  = dict()
            if isLast:
                weights_bias['coder_w'] = tf.get_variable(name=pre_name+"_coder_w",initializer=xavier_init(n_output,n_input), regularizer=self.l2_reg)
                weights_bias['s_coder_w'] = tf.get_variable(name=pre_name+"_s_coder_w",initializer=xavier_init(n_output,s_input), regularizer=self.l2_reg)
                weights_bias['coder_b'] = tf.get_variable(name=pre_name+"_coder_b",initializer=tf.truncated_normal([n_input],dtype=tf.float32))
                weights_bias['s_coder_b'] = tf.get_variable(name=pre_name+"_s_coder_b",initializer=tf.truncated_normal([s_input],dtype=tf.float32))

#                 weights_bias['coder_w'] = tf.Variable(xavier_init(n_output,n_input))
#                 tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(self.l2_loss_weight)(weights_bias['coder_w']))
#                 weights_bias['s_coder_w'] = tf.Variable(xavier_init(n_output,s_input))
#                 tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(self.l2_loss_weight)(weights_bias['s_coder_w']))
#                 weights_bias['coder_b'] = tf.Variable(tf.truncated_normal([n_input],dtype=tf.float32))
#                 tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(self.l2_loss_weight)(weights_bias['coder_b']))
#                 weights_bias['s_coder_b'] = tf.Variable(tf.truncated_normal([s_input],dtype=tf.float32))
#                 tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(self.l2_loss_weight)(weights_bias['s_coder_b']))
            else:
                weights_bias['coder_w'] = tf.get_variable(name=pre_name+"_coder_w",initializer=xavier_init(n_input,n_output), regularizer=self.l2_reg)
                weights_bias['s_coder_w'] = tf.get_variable(name=pre_name+"_s_coder_w",initializer=xavier_init(s_input,n_output), regularizer=self.l2_reg)
                weights_bias['coder_b'] = tf.get_variable(name=pre_name+"_coder_b",initializer=tf.truncated_normal([n_output],dtype=tf.float32))

#                 weights_bias['coder_w'] = tf.Variable(xavier_init(n_input,n_output))
#                 tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(self.l2_loss_weight)(weights_bias['coder_w']))
#                 weights_bias['s_coder_w'] = tf.Variable(xavier_init(s_input,n_output))
#                 tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(self.l2_loss_weight)(weights_bias['s_coder_w']))
#                 weights_bias['coder_b'] = tf.Variable(tf.truncated_normal([n_output],dtype=tf.float32))
#                 tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(self.l2_loss_weight)(weights_bias['coder_b']))
        return weights_bias    
        
    def add_noise(self, x_):
        #添加高斯噪声
        n = np.random.normal(0, self.scale, (len(x_), len(x_[0])))
        return x_ + n.astype(np.float32)
        
