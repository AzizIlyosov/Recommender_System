import tensorflow as tf
import numpy as np

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    high = constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform([fan_in,fan_out], minval=low, maxval=high, dtype=tf.float32)

def mse_by_part(x,y,s_size,alpha):
    assert (0 <= alpha <= 1),'alpha must set between 0 and 1.'
    result1 = tf.reduce_mean(tf.reduce_mean(tf.pow(tf.subtract(x[:,:s_size], y[:,:s_size]), 2.0), axis=1))
    result2 = tf.reduce_mean(tf.reduce_mean(tf.pow(tf.subtract(x[:,s_size:], y[:,s_size:]), 2.0), axis=1))
    return alpha * result1 + (1-alpha)*result2  

def mse_mask(x,y):
    # x=0的地方，不计算mse,用的sign,>0的地方置1，=0的地方置0
    mask = tf.sign(tf.abs(x))
    # return tf.reduce_mean(tf.reduce_sum(mask * tf.pow(tf.subtract(x,y),2.0)))
    return tf.reduce_mean(tf.reduce_sum(mask * tf.pow(tf.subtract(x, y), 2.0),axis=1))

def rmse_mask(x,y):
    # x=0的地方，不计算mse,用的sign,>0的地方置1，=0的地方置0
    mask = tf.sign(tf.abs(x))
    num = tf.reduce_sum(mask)
    mse = tf.reduce_mean(tf.reduce_sum(mask * tf.pow(tf.subtract(x, y), 2.0)))
    return tf.sqrt(mse/num)

class MLPModel(object):
    def __init__(self, sess, Rshape, Ushape, Ishape, dims=[8,6,4], activation="sigmoid", learning_rate=0.01,
                 n_epoch =1000, print_step=500,batch_size=100, is_training = True, reg_lambda=0.0, alpha=1, beta=1, 
                 delta=1, noise_rate=0):
        self.sess = sess
        self.Rshape = Rshape
        self.Ushape = Ushape
        self.Ishape = Ishape
        self.dims = dims
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.print_step = print_step
        self.batch_size = batch_size
        self.is_training = is_training
        self.reg_lambda = reg_lambda
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.noise_rate = noise_rate
        self.regularizer = tf.contrib.layers.l2_regularizer
#         tf.nn.l2_loss
#         tf.contrib.layers.l2_regularizer
        
    def encoder(self, inputs, units, noise_rate, layerlambda, name="encoder"):
        input_size = int(inputs.shape[1]) # 输入的维度
        with tf.variable_scope(name):
            # 添加mask噪声
            corrupt = tf.layers.dropout(inputs, noise_rate, training=self.is_training)
            en_w = tf.get_variable(name="encoder_w",initializer=xavier_init(input_size,units),
                                   regularizer=self.regularizer(layerlambda))
            en_b = tf.get_variable('encoder_b',shape=[1,units],initializer=tf.constant_initializer(0.0),dtype=tf.float32,
                                   regularizer=self.regularizer(layerlambda))
            
            tensor = tf.add(tf.matmul(corrupt, en_w), en_b)
            encoded = self.activate(tensor)
#             encoded = self.activate(tf.layers.batch_normalization(tensor))
            
            self.en_w = en_w
            self.en_b = en_b
            
        return encoded
    
    def decoder(self, inputs, units, layerlambda, name="encoder"):
        input_size =int(inputs.shape[1])
        with tf.variable_scope(name):
            de_w = tf.get_variable(name="decoder_w",initializer=xavier_init(input_size,units),
                                   regularizer=self.regularizer(layerlambda))
            de_b = tf.get_variable('decoder_b',shape=[1,units],initializer=tf.constant_initializer(0.0),dtype=tf.float32,
                                   regularizer=self.regularizer(layerlambda))
            
            decoded = tf.add(tf.matmul(inputs, de_w), de_b)
            act_decoded = self.activate(decoded)
#             act_decoded = self.activate(tf.layers.batch_normalization(decoded))
            self.de_w = de_w
            self.de_b = de_b
            
        return decoded, act_decoded
    
    def create_graph(self):
        self.R = tf.placeholder(tf.float32,None,name="Rating_Matrix")
        # 用户网络
        input_size = self.Rshape[1] + self.Ushape[1]
        self.u_x = tf.placeholder(tf.float32, [None, input_size], name="user_input")
        # ------encoder------
        loss_name = "loss_U"
        self.U_enc_layers = []
        input_data = self.u_x
        u_info = self.u_x[:,self.Ushape[1]:input_size]
        for i in range(len(self.dims)):
            layer_name = "U_encoder_layer"+str(i+1)
            out = self.encoder(input_data, self.dims[i], self.noise_rate, self.reg_lambda, layer_name)
            self.U_enc_layers.append(out)
            reg_losses = tf.losses.get_regularization_losses(layer_name)
            input_data = tf.concat([out,u_info],axis=1)
#             input_data = tf.concat([out,self.u_x],axis=1)
        self.U = out
        for loss in reg_losses:
            tf.add_to_collection(loss_name, loss)
        # ------decoder------
        decode_dims = list(self.dims[:len(self.dims)-1])
        decode_dims.reverse()
        decode_dims.append(input_size)
        self.U_dec_layers = []
        input_data = self.U
        for i in range(len(decode_dims)):
            layer_name = "U_decoder_layer"+str(i+1)
            decoded, act_decoded = self.decoder(input_data, decode_dims[i], self.reg_lambda, layer_name)
            self.U_dec_layers.append(act_decoded)
            reg_losses = tf.losses.get_regularization_losses(layer_name)
            input_data = tf.concat([act_decoded,u_info],axis=1)
#             input_data = tf.concat([act_decoded,self.u_x],axis=1)
        self.U_pred = decoded
        for loss in reg_losses:
            tf.add_to_collection(loss_name, loss)
        # ------loss------
        self.reg_loss_u = tf.add_n(tf.get_collection(loss_name))
        self.pred_loss_u = mse_by_part(self.U_pred, self.u_x, self.Rshape[1], self.alpha)
        self.loss_u = self.reg_loss_u+self.pred_loss_u
        
        # 物品网络
        input_size = self.Rshape[0] + self.Ishape[1]
        self.i_x = tf.placeholder(tf.float32, [None, input_size], name="item_input")
        # ------encoder------
        loss_name = "loss_I"
        self.I_enc_layers = []
        input_data = self.i_x
        i_info = self.i_x[:,self.Ishape[1]:input_size]
        for i in range(len(self.dims)):
            layer_name = "I_encoder_layer"+str(i+1)
            out = self.encoder(input_data, self.dims[i], self.noise_rate, self.reg_lambda, layer_name)
            self.I_enc_layers.append(out)
            reg_losses = tf.losses.get_regularization_losses(layer_name)
            input_data = tf.concat([out,i_info],axis=1)
#             input_data = tf.concat([out,self.i_x],axis=1)
        self.V = out
        for loss in reg_losses:
            tf.add_to_collection(loss_name, loss)
        # ------decoder------
        decode_dims = list(self.dims[:len(self.dims)-1])
        decode_dims.reverse()
        decode_dims.append(input_size)
        self.I_dec_layers = []
        input_data = self.V
        for i in range(len(decode_dims)):
            layer_name = "I_decoder_layer"+str(i+1)
            decoded, act_decoded = self.decoder(input_data, decode_dims[i], self.reg_lambda, layer_name)
            self.I_dec_layers.append(act_decoded)
            reg_losses = tf.losses.get_regularization_losses(layer_name)
            input_data = tf.concat([act_decoded,i_info],axis=1)
#             input_data = tf.concat([act_decoded,self.i_x],axis=1)
        self.I_pred = decoded
        for loss in reg_losses:
            tf.add_to_collection(loss_name, loss)
        # ------loss------
        self.reg_loss_i = tf.add_n(tf.get_collection(loss_name))
        self.pred_loss_i = mse_by_part(self.I_pred, self.i_x, self.Rshape[0], self.alpha)
        self.loss_i = self.reg_loss_i+self.pred_loss_i  # I的总误差
        
        # 计算模型总误差
        self.R_pred = tf.matmul(self.U, tf.transpose(self.V))
        self.pred_loss = mse_mask(self.R, self.R_pred ) #矩阵的误差
        reg_loss_u_and_i = tf.reduce_mean(tf.norm(self.U,axis=1))+tf.reduce_mean(tf.norm(self.V,axis=1))
        self.loss = self.pred_loss + self.reg_lambda*reg_loss_u_and_i+\
                    self.beta * self.loss_u + self.delta * self.loss_i
        self.rmse = rmse_mask(self.R, self.R_pred)


    def train(self, load_data_func):
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
#         self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(self.n_epoch):
            if self.batch_size is None:
                n_batch = 1
            else:
                n_batch = self.Rshape[0]//self.batch_size
            Rfile = "./Data/rating.npy"
            data_generator = load_data_func(Rfile,n_batch, batch_size=self.batch_size, shuffle=False)
            for _ in range(n_batch):
                batch_u, batch_i, batch_R = next(data_generator)
                feed_data={self.u_x:batch_u, self.i_x:batch_i, self.R: batch_R}
                _, loss, rmse = self.sess.run([self.optimizer, self.loss, self.rmse],
                                                    feed_dict=feed_data)
            if (epoch +1)%self.print_step == 0:
                print("epoch ",(epoch+1)," train loss: ", loss,"rmse: ",rmse)
        data_generator = load_data_func(Rfile,0, batch_size=None, shuffle=False)
        batch_u, batch_i, batch_R = next(data_generator)
        feed_data={self.u_x:batch_u, self.i_x:batch_i, self.R: batch_R}
        return self.sess.run([self.U, self.V], feed_dict=feed_data)
    
    def activate(self, x):
        #激活神经元
        if self.activation == "sigmoid":
            return tf.nn.sigmoid(x)
        if self.activation == "tanh":
            return tf.nn.tanh(x)
        if self.activation == "relu":
            return tf.nn.relu(x)
        else:
            print("Invalid activation!")
            return x