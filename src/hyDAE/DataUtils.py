import numpy as np
from sklearn import preprocessing

def data_generator_false(nb_batch,batch_size=None,shuffle = True):
    R = np.array([[0,0.9,0.9,0.7,0.6,0.7,0.9,0.6,0.2,0.2],[0.6,0.4,0.9,0.7,0.7,0.2,0,0.4,0.6,0.7],[0.4,0.5,0.5,0.9,0.7,0.8,0.9,0.7,0.8,0],[0.1,0.5,0.2,0.8,0.9,0,0.6,0.8,0.3,0.3]])
    U = np.array([[0,1,0,1,0],[1,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1]])
    I = np.array([[1,0,1,1],
                  [0,0,1,0],
                  [0,0,1,1],
                  [1,1,1,1],
                  [1,0,0,0],
                  [1,0,0,0],
                  [1,1,1,1],
                  [1,1,1,1],
                  [0,0,1,1],
                  [0,1,0,1]])
    
    if shuffle:
        ru = np.random.permutation(U.shape[0])      # 只在第一次读的时候做shuffle
        U = U[ru,:]
        ri = np.random.permutation(I.shape[0])
        I = I[ri,:]
    else:
        ru = range(U.shape[0])
        ri = range(I.shape[0])

    if batch_size is None:
        R = R[ru,:]
        R = R[:,ri]
        batch_U = np.concatenate((R, U), axis=1)
        batch_I = np.concatenate((R.T, I), axis=1)                                  # 转置
        yield batch_U, batch_I, R
    else:
        batch = 0
        while batch <= nb_batch:
            batch_U = U[batch*batch_size:(batch+1)*batch_size]
            batch_I = I[batch*batch_size:(batch+1)*batch_size]
            batch_R_u = R[ru,:][batch*batch_size:(batch+1)*batch_size]            # 所选用户对应的评分项
            batch_R_i = R[:,ri][:,batch*batch_size:(batch+1)*batch_size]
            batch_R = batch_R_u[:,ri][:,batch*batch_size:(batch+1)*batch_size]
            batch_U = np.concatenate((batch_R_u,batch_U),axis=1)
            batch_I = np.concatenate((batch_R_i.T,batch_I),axis=1)                  # 转置
            batch += 1
            yield batch_U,batch_I,batch_R
            
def data_generator(nb_batch,batch_size=None,shuffle = True):
    U = np.load('./Data/user.npy',mmap_mode='r')
    I = np.load('./Data/item.npy',mmap_mode='r')
    R = np.load('./Data/u1_train.npy',mmap_mode='r')
    
    if shuffle:
        ru = np.random.permutation(U.shape[0])      # 只在第一次读的时候做shuffle
        U = U[ru,:]
        ri = np.random.permutation(I.shape[0])
        I = I[ri,:]
    else:
        ru = range(U.shape[0])
        ri = range(I.shape[0])

    if batch_size is None:
        R = R[ru,:]
        R = R[:,ri]
        batch_U = np.concatenate((R, U), axis=1)
        batch_I = np.concatenate((R.T, I), axis=1)                                  # 转置
        yield batch_U, batch_I, R
    else:
        batch = 0
        while batch <= nb_batch:
            batch_U = U[batch*batch_size:(batch+1)*batch_size]
            batch_I = I[batch*batch_size:(batch+1)*batch_size]
            batch_R_u = R[ru,:][batch*batch_size:(batch+1)*batch_size]            # 所选用户对应的评分项
            batch_R_i = R[:,ri][:,batch*batch_size:(batch+1)*batch_size]
            batch_R = batch_R_u[:,ri][:,batch*batch_size:(batch+1)*batch_size]
            batch_U = np.concatenate((batch_R_u,batch_U),axis=1)
            batch_I = np.concatenate((batch_R_i.T,batch_I),axis=1)                  # 转置
            batch += 1
            yield batch_U,batch_I,batch_R

def createRatingMatrix(filename, Rshape):
    ratings_matrix = np.zeros(Rshape)
    f = open(file=filename)
    line = f.readline()
    while line != "":
        try:
            content = line.split("\t")
            user_id = int(content[0])
            item_id = int(content[1]) 
            rating = int(content[2])
            ratings_matrix[user_id-1][item_id-1] = rating
        except:
            break
        line = f.readline()
    f.close()
#     print(ratings_matrix[0])
#     ratings_matrix = ratings_matrix/5.0
#     print(ratings_matrix.shape)
#     print(ratings_matrix[0])
    return ratings_matrix

def createItemMatrix(filename, Ishape):
    # id_19个种类
    item_matrix = np.zeros(Ishape)
    f = open(file=filename,encoding="utf-16")
    line = f.readline()
    while line != "":
        try:
            content = line.split("|")
            item_info = []
            item_info.append(content[0]) 
            item_info.extend(content[5:-1])
            item_info.append(content[-1].split('\n')[0])
            try:
                item_info = [int(i)for i in item_info]
            except:
                print (item_info[0])
            item_matrix[item_info[0]-1] = np.array(item_info)
        except:
            break
        line = f.readline()
    f.close()
    item_matrix = item_matrix[:,1:] #去除id这一列
    return item_matrix

def createUserMatrix(filename, Ushape, occupation_list):
    # id_年纪_性别_21种职业
    user_matrix = np.zeros(Ushape)
    f = open(file=filename)
    line = f.readline()
    while line != "":
        try:
            content = line.split("|")
            user_info = []
            user_info.append(content[0]) # id
            user_info.append(content[1]) # age
            if content[2] == "M":
                user_info.append(0)
            else:
                user_info.append(1)
            occupation_one_hot = [content[3]==oc for oc in occupation_list]
            user_info.extend(occupation_one_hot)
            try:
                user_info = [int(i)for i in user_info]
            except:
                print (user_info[0])
            user_matrix[user_info[0]-1] = np.array(user_info)
        except:
            break
        line = f.readline()
    f.close()
#     print(user_matrix[0])
    user_matrix = user_matrix[:,1:] #去除id这一列
    user_matrix[:,0] = preprocessing.maxabs_scale(user_matrix[:,0])   # 归一化处理age
#     print(user_matrix[0])
    return user_matrix

def createBaseInfo(filename, occupation_file):
    f = open(file=filename) #读取数据集信息
    line = f.readline()
    content = line.split(' ')
    num_u = int(content[0]) # 用户数
    line = f.readline()
    content = line.split(' ')
    num_i = int(content[0]) # 物品数
    f = open(file=occupation_file) # 读取职业信息
    oc_list = f.readlines()
    oc_list =[occ.split('\n')[0] for occ in oc_list]
    f.close()
    return num_u, num_i, oc_list

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf) 
    path = '../../../SDAE-recommendation-master/SDAE-recommendation-master/data/ml-100k/'
    num_users,num_items,occup_list = createBaseInfo(path+'u.info',path+'u.occupation')
    print(num_users,num_items,occup_list)
    UM = createUserMatrix(path+'u.user',(num_users,(len(occup_list)+3)), occup_list)
    np.save("./Data/user", UM)
    IM = createItemMatrix(path+'u.item',(num_items,20))
    np.save("./Data/item", IM)
    RM = createRatingMatrix(path+'u.data',(num_users,num_items))
    np.save("./Data/rating", RM)
    RM_base = createRatingMatrix(path+'u1.base',(num_users,num_items))
    np.save("./Data/u1_train", RM_base)
    RM_test = createRatingMatrix(path+'u1.test',(num_users,num_items))
    np.save("./Data/u1_test", RM_test)
    print(UM.shape)
    print(IM.shape)
    print(RM_base.shape)
    print(RM_test.shape)


