import numpy as np
from DAE import DenoisingAutoEncoder
from numpy import float32

if __name__ == '__main__':
    
    model = DenoisingAutoEncoder(optimizer="adam",activation="sigmoid",loss="mse")
    c = np.random.rand(6,10).astype(float32)
    s_u = np.random.rand(6,15).astype(float32)
    print(c)
    print("\n============\n")
    print(s_u)
    print("\n============\n")
    
    x_ = model.train(x_=c, s_x_=s_u,dims=[10,8,6,4],epoches=10000,batch_size=2,print_step=10000)
    print(x_)
