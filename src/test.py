import numpy as np
from hybrid_aSDAE import DenoisingAutoEncoder

if __name__ == '__main__':
    
    model = DenoisingAutoEncoder(optimizer="adam",activation="sigmoid",loss="mse")
#     c = np.random.rand(6,10).astype(np.float32)
#     s_u = np.random.rand(6,15).astype(np.float32)
#     s_v = np.random.rand(10,18).astype(np.float32)
    
    c = np.array([[1,0,6],[3,4,8],[9,5,6],[9,0,2]])
    s_u = np.array([[1,2,3,4],[9,5,8,0],[0,0,1,9],[0,1,3,0]])
    s_v = np.array([[2,0],[7,8],[0,7]])
    
    
    
#     print(c)
#     print("\n============\n")
#     print(s_u)
#     print("\n============\n")
#     print(s_v)

    U, V, pred_u=  model.train(user_item_ratings_clean=c,user_info_clean=s_u, item_info_clean=s_v,dims=[6,4,2,4,6])
    print(U)
    print("\n============\n")
    print(V) 
    print("\n============\n")
    print(pred_u) 
    print("\n============\n")
    print(c)  
    print("\n============\n")
    print(np.dot(U,np.transpose(V)))   
