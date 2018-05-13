import numpy as np
import tensorflow as tf
from hyDAE.DataUtils import data_generator, data_generator_false, calculate_recall,recommend_result
from hyDAE.MLPModel import MLPModel

def test_false_data():
    args = {
        "noise_rate"     : 0.0,
        "dims"   : (10,8,6,4),
        "learning_rate": .0004,
        "n_epoch"  : 5000,
        "print_step": 5000,
        "batch_size": None,
        "reg_lambda":0.01,
        "alpha"     :0.2,
        "beta"      :0.8,
        "delta"     :1,
    }
    tf.reset_default_graph()
    R = np.array([[0,0.9,0.9,0.7,0.8,0.7,0,0,0.2,0.2],[0.6,0.4,0.9,0.7,0.7,0.2,0,0.4,0.6,0.7],[0.4,0.5,0.5,0.9,0.7,0.8,0.9,0.7,0.8,0],[0.1,0.5,0.2,0.8,0.9,0,0.6,0.8,0.3,0.3]])
    with tf.Session() as sess:
        print("Initializing...")
        model = MLPModel(sess,(4, 10),(4, 5),(10, 4),is_training = True,**args)
        print("build model...")
        model.create_graph()
        print("training...")
        U, V= model.train(load_data_func=data_generator_false)
        print(R,"\n")
        print(np.dot(U, V.T))
        print(calculate_recall(R,np.dot(U, V.T)))
#         print(R)
        ratings, ids = recommend_result(R,np.dot(U, V.T),K=1)
        print(ratings[0], ids[0])

def test_true_data():
    args = {
        "noise_rate"     : 0.003,
        "dims"   : (512,256,128,64),
        "learning_rate": .0004,
        "n_epoch"  : 100,
        "print_step": 50,
        "batch_size": None,
        "reg_lambda":0.01,
        "alpha"     :0.2,
        "beta"      :0.8,
        "delta"     :1,
    }
    tf.reset_default_graph()
   
    with tf.Session() as sess:
        print("Initializing...")
        model = MLPModel(sess,(943, 1682),(943, 23),(1682, 19),is_training = True,**args)
        print("build model...")
        model.create_graph()
        print("training...")
        U, V= model.train(load_data_func=data_generator)
        R = np.load('./Data/rating.npy',mmap_mode='r')
        pred = np.dot(U, V.T)
        print("\n", R[0])
        print(pred[0])
       
        print(calculate_recall(R,pred,K=(50,100,150,200)))
        ratings, ids = recommend_result(R,pred,K=10)
        print(ratings[0], ids[0])

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
#     test_false_data()
    test_true_data()