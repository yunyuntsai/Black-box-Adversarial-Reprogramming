import tensorflow as tf
import numpy as np




def freq_idx_search(NonAsd, Asd, mlm_num):


    nonasd_value = 0
    asd_value = 0

    for i in range(1000):
        if NonAsd[i][1] != 0:
            NonAsd[i][0] = (NonAsd[i][0] /  NonAsd[i][1]) 
            nonasd_value += NonAsd[i][0]
        if Asd[i][1] != 0:
            Asd[i][0] = (Asd[i][0] / Asd[i][1]) 
            asd_value += Asd[i][0]

    sort_NonAsd =   NonAsd[NonAsd[:,0].argsort()[::-1]] 
    NonAsd_index = NonAsd[:,0].argsort()[::-1][:30]

    sort_Asd = Asd[Asd[:,0].argsort()[::-1]]
    Asd_index = Asd[:,0].argsort()[::-1][:30]

    Asd_mlm = []
    NonAsd_mlm = []

    for i in range(30):
        if NonAsd_index[i] == Asd_index[i]:
            if sort_NonAsd[i,0] > sort_Asd[i, 0]:
                NonAsd_mlm.append(NonAsd_index[i])
        else:
            if NonAsd_index[i] in Asd_index:
                    k = np.where(Asd_index == NonAsd_index[i])
                    if i < k[0]:
                        NonAsd_mlm.append(NonAsd_index[i])
            else:
                NonAsd_mlm.append(NonAsd_index[i])
    for i in range(30):
        if Asd_index[i] == NonAsd_index[i]:
            if sort_Asd[i, 0] > sort_NonAsd[i, 0]:
                Asd_mlm.append(Asd_index[i])
        else: 
            if Asd_index[i] in NonAsd_index:
                k = np.where(NonAsd_index == Asd_index[i])
                if i < k[0]:
                    Asd_mlm.append(Asd_index[i])
            else:
                Asd_mlm.append(Asd_index[i])

    number_for_mapping = np.minimum(len(Asd_mlm), len(NonAsd_mlm))
    # if number_for_mapping<6:
    #     self.mlm_num = number_for_mapping

    mlm_index = np.array([NonAsd_mlm[:mlm_num], Asd_mlm[:mlm_num]]).ravel()
    mlm_index =  np.array(mlm_index)

    return mlm_index

def freq_mapping(imageprob, mlm_index):

    batch_size = 10
    q_batch = 1
    mlm_num = 6

    mlm = mlm_index[:]
    freq_prob = tf.gather(imageprob, mlm, axis=1)
    freq_reshape_prob = tf.reshape(freq_prob, shape=[batch_size*(q_batch+1), mlm_num, 2])
    multi_freq_prob = tf.reduce_mean(freq_reshape_prob, axis=1)

    return multi_freq_prob

def label_mapping():

    imagenet_label = np.zeros([1000,10])
    imagenet_label[0:10,0:10]=np.eye(10)

    return tf.constant(imagenet_label, dtype=tf.float32) 

def multi_prob(disturbed_prob):

    batch_size = 10
    q_batch = 1
    mlm_num = 6
    reshape_prob = tf.reshape(disturbed_prob, shape=[batch_size*(q_batch+1), mlm_num, 2])
    multi_label_prob = tf.reduce_mean(reshape_prob, axis=1)

    return multi_label_prob