import numpy as np
import tensorflow as tf
from numpy import float32

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    high = constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform([fan_in,fan_out], minval=low, maxval=high, dtype=tf.float32)

class DenoisingAutoEncoder(object):
    def __init__(self, lr = 0.01, alpha = 0.1, activation="sigmoid", optimizer="adam", loss="mse", scale = 0.1):
#       lr：学习率
#       rating_loss_weight：平衡参数
#       activation：激活函数
#       optimizer：优化器
#       scale：高斯噪声的标准差
        self.lr = lr
        self.alpha = alpha
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.scale = scale
        self.weights_bias = []
        self.s_weights_bias = []
    def _init_s_weights_biaes(self, n_input, n_output):
        weights_biaes  = dict()
        weights_biaes['coder_w'] = tf.Variable(xavier_init(n_input,n_output))
        weights_biaes['coder_b'] = tf.Variable(tf.truncated_normal([n_output],dtype=tf.float32))

        return weights_biaes   
    def _init_weights_biaes(self, n_input, n_output):
        #权重与偏置初始化
        weights_biaes  = dict()
        weights_biaes['encoder_w'] = tf.Variable(xavier_init(n_input,n_output))
        weights_biaes['encoder_b'] = tf.Variable(tf.truncated_normal([n_output],dtype=tf.float32))
        weights_biaes['decoder_w'] = tf.Variable(tf.transpose(weights_biaes['encoder_w']))
        weights_biaes['decoder_b'] = tf.Variable(tf.truncated_normal([n_input],dtype=tf.float32))

        return weights_biaes
    
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
        
    def encoder(self, x, weights_bias, name=None):
        #编码器         
        tensor = tf.add(tf.matmul(x,weights_bias['encoder_w']),weights_bias['encoder_b'], name)
        return self.activate(tensor, name)
    
    def decoder(self, x, weights_bias=None, s_weights_bias=None,name=None):
        #解码器
        if s_weights_bias is not None:
#             print(x)
#             print(s_weights_bias['coder_w'])
#             print(s_weights_bias['coder_b'])
            tensor = tf.add(tf.matmul(x,tf.transpose(s_weights_bias['coder_w'])),tf.transpose(s_weights_bias['coder_b']), name)
        else:
            tensor = tf.add(tf.matmul(x,weights_bias['decoder_w']),weights_bias['decoder_b'], name)
        return self.activate(tensor, name)
    
    def encoder_all(self, x, s, weights_bias, s_weights_bias, name=None):
        tensor = tf.add(tf.matmul(x,weights_bias['encoder_w']),weights_bias['encoder_b'])
        s_tensor = tf.add(tf.matmul(s,s_weights_bias['encoder_w']),s_weights_bias['encoder_b'])
        tensor_all = tf.add(tensor, s_tensor)
        return self.activate(tensor_all, name)
    
    def decoder_all(self, x, s, weights_bias, s_weights_bias, name=None):
        print(x)
        print(s)
        print(s_weights_bias['coder_w'])
        print(weights_bias['decoder_w'])
        s_tensor = tf.add(tf.matmul(s,s_weights_bias['coder_w']),s_weights_bias['coder_w'])
        tensor = tf.add(tf.matmul(x,weights_bias['decoder_w']),weights_bias['decoder_b'], name)
        print(s_tensor)
        print(tensor)
        tensor_all = tf.add(tensor, s_tensor)
        print("---： ", tensor_all)
        return self.activate(tensor_all, name)
    
    def add_noise(self, x_):
        #添加高斯噪声
        n = np.random.normal(0, self.scale, (len(x_), len(x_[0])))
        return x_ + n.astype(float32)
    
    
    def train(self, x_,s_x_=None, dims=[512,256,128,64], epoches=500, batch_size=100,print_step=100):
        depth = len(dims)-1
        if s_x_ is not None:
            s_temp = np.copy(s_x_)
            s_temp_x = self.add_noise(s_temp)
        else:
            s_temp_x = None
            
        for i in range(depth):
            temp = np.copy(x_)
            if i == 0:
                temp_x = self.add_noise(temp)
            else:
                temp_x = temp
            if s_x_ is None:
                x_=self.fit_one(x_=x_, x=temp_x, n_output=dims[i+1], 
                                epoches=epoches, batch_size=batch_size, print_step=print_step)
            else:
                x_ = self.fit_two(x_=x_, x=temp_x, s_x_=s_x_, s_x=s_temp_x, n_output=dims[i+1], 
                           epoches=epoches, batch_size=batch_size, print_step=print_step)
#         self.save_model() #模型参数保存
        return x_  
    
    def predict(self, x):
        tf.reset_default_graph()
        # 加载模型参数
        saver = tf.train.import_meta_graph("model_save\\model.ckpt.meta") 
        
        with tf.Session() as sess:
            saver.restore(sess, "model_save\\model.ckpt")
#             saver.restore(sess, tf.train.latest_checkpoint('model_save\\'))
#             all_vars_w = tf.get_collection('vars_w')
#             all_vars_b = tf.get_collection('vars_b')
#             print(all_vars_w)
#             print(all_vars_b)
#             for v in all_vars_w:
#                 print(v)
#                 print(v.name)
#                 v_ = v.eval() # sess.run(v)
#                 print(v_)
            weights = []
            weights.append(sess.run(tf.get_default_graph().get_tensor_by_name('layer_w_1:0')))
            weights.append(sess.run(tf.get_default_graph().get_tensor_by_name('layer_w_2:0')))
            weights.append(sess.run(tf.get_default_graph().get_tensor_by_name('layer_w_3:0')))
            bias = []
            bias.append(sess.run(tf.get_default_graph().get_tensor_by_name('layer_b_1:0')))
            bias.append(sess.run(tf.get_default_graph().get_tensor_by_name('layer_b_2:0')))
            bias.append(sess.run(tf.get_default_graph().get_tensor_by_name('layer_b_3:0')))
             
            for i in range(len(weights)):
                layer = tf.add(tf.matmul(x,weights[i]),bias[i])
                x = self.activate(layer)
            return x.eval(session=sess)
        
    def fit_one(self, x_, x, n_output, epoches, batch_size, print_step):
        # 评分数据
        n_input = len(x_[0])
        p_x_ = tf.placeholder(tf.float32, [None, n_input],name="1") #原始数据
        p_x = tf.placeholder(tf.float32, [None, n_input],name="2")  #噪声数据
        
        weights_bias = self._init_weights_biaes(n_input,n_output)
        
        encode_x = self.encoder(p_x,weights_bias)
        decode_x = self.decoder(encode_x,weights_bias)
        
        if self.loss == "mse":        
            loss = tf.reduce_mean(tf.square(tf.subtract(p_x_, decode_x)))
        elif self.loss == "cross_entropy":
            loss = -tf.reduce_mean(tf.reduce_sum(p_x_ * tf.log(decode_x)+(1-p_x_)*tf.log(1-decode_x),1))
        if self.optimizer == "adam":
            train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        else:
            train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epoches):
                batches = self.get_batches(x_, x, batch_size)
                for batch in batches:
                    b_x_, b_x = zip(*batch)
                    _ ,c = sess.run((train_op,loss), feed_dict={p_x_: b_x_, p_x:b_x})
                if (epoch+1) % print_step == 0:
                    print("With One---Epoch:", '%d' %(epoch+1),"output_dim:",'%d' % n_output,"loss=","{:.9f}".format(c))
            self.weights_bias.append(sess.run(weights_bias["encoder_w"]))
            self.weights_bias.append(sess.run(weights_bias["encoder_b"]))
            return sess.run(encode_x, feed_dict={p_x_:x_, p_x:x})

    def fit_two(self, x_, x, s_x_, s_x, n_output, epoches, batch_size, print_step):
        if s_x_ is None:
            self.fit_x(x_,x,n_output, epoches, batch_size, print_step)
        else:
            n_input = len(x_[0])
            s_n_input = len(s_x_[0])
            
            p_x_ = tf.placeholder(tf.float32, [None, n_input],name="1") #原始数据
            p_x = tf.placeholder(tf.float32, [None, n_input],name="2")  #噪声数据
            s_p_x_ = tf.placeholder(tf.float32, [None, s_n_input],name="3") #原始数据
            s_p_x = tf.placeholder(tf.float32, [None, s_n_input],name="4")  #噪声数据
            
            weights_bias = self._init_weights_biaes(n_input,n_output)
            s_weights_bias = self._init_weights_biaes(s_n_input,n_output)
            
            encode_all = self.encoder_all(p_x, s_p_x, weights_bias,s_weights_bias)
            decode_all_x = self.decoder(encode_all,weights_bias) #解码x
            s_decode_all_x = self.decoder(encode_all,s_weights_bias)#解码side_info
             
            if self.loss == "mse":        
                cost = tf.reduce_mean(tf.square(tf.subtract(p_x_, decode_all_x)))
                s_cost = tf.reduce_mean(tf.square(tf.subtract(s_p_x_, s_decode_all_x)))
            elif self.loss == "cross_entropy":
                cost = -tf.reduce_mean(tf.reduce_sum(p_x_ * tf.log(decode_all_x)+(1-p_x_)*tf.log(1-decode_all_x),1))
                s_cost = -tf.reduce_mean(tf.reduce_sum(s_p_x_ * tf.log(s_decode_all_x)+(1-s_p_x_)*tf.log(1-s_decode_all_x),1))
            loss = cost + s_cost
#             loss = self.alpha*cost + (1.0 - self.alpha)*s_cost
            if self.optimizer == "adam":
                train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
            else:
                train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)
            
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for epoch in range(epoches):
                    b_x_, b_x, s_b_x_, s_b_x = self.get_all_batches(x_, x, s_x_, s_x, batch_size)
                    for i in range(len(b_x_)):
                        _ ,c= sess.run((train_op,loss), feed_dict={p_x_: b_x_[i], p_x:b_x[i], s_p_x_:s_b_x_[i], s_p_x:s_b_x[i]})
                    if (epoch+1) % print_step == 0:
                        print("With Two---Epoch:", '%d' %(epoch+1),"output_dim:",'%d' % n_output,"loss=","{:.9f}".format(c))
                self.weights_bias.append(sess.run(weights_bias["encoder_w"]))
                self.weights_bias.append(sess.run(weights_bias["encoder_b"]))
                self.s_weights_bias.append(sess.run(s_weights_bias["encoder_w"]))
                self.s_weights_bias.append(sess.run(s_weights_bias["encoder_b"]))
                return sess.run(encode_all, feed_dict={p_x_:x_, p_x:x, s_p_x_:s_x_, s_p_x:s_x})
    
    def get_batches(self, x_, x, size):
        #随机挑选批量数据
        shuff = list(zip(x_, x)) 
#         np.random.shuffle(shuff)
        batches = [_ for _ in self.get_batch(shuff, size)]
        return batches
 
    def get_all_batches(self, x_, x, s_x_, s_x, size):
        inx = np.random.choice(len(x_),len(x_),replace=False)
        batches_x_ = []
        batches_x = []
        s_batches_x_ = []
        s_batches_x = []
        for i in range(0, len(inx), size):
            batches_x_.append(x_[inx[i:i+size]])
            batches_x.append(x[inx[i:i+size]])
            s_batches_x_.append(s_x_[inx[i:i+size]])
            s_batches_x.append(s_x[inx[i:i+size]])
        return batches_x_, batches_x, s_batches_x_, s_batches_x
    
    def get_batch(self, data, size):
        data = np.array(data)
        for i in range(0, data.shape[0], size):
            yield data[i:i+size]
    
    def save_model(self):
        # 模型参数保存
        weights_biaes = dict()
        for i in range(0,len(self._weights_biaes),2):
            num_layer = int((i+2)/2)
            w = tf.Variable(self._weights_biaes[i],name="layer_w_"+str(num_layer)) 
            b = tf.Variable(self._weights_biaes[i+1],name="layer_b_"+str(num_layer)) 
            weights_biaes["layer_w_"+str(num_layer)] = w
            weights_biaes["layer_b_"+str(num_layer)] = b
            tf.add_to_collection('vars_w', w)
            tf.add_to_collection('vars_b', b)
            
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, 'model_save\\model.ckpt') 