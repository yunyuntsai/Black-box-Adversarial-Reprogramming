import numpy as np



def get_key(filename):
    f_split = filename.split('_')
    if f_split[3] == 'rois':
        key = '_'.join(f_split[0:3]) 
    else:
        key = '_'.join(f_split[0:2])
    return key

def get_label(filename, labels):
    assert (filename in labels)
    return labels[filename]


def process_data(corr_data, sample_list):

    corr_mat = []
    onehot_mat = []
        
    for index in range(len(sample_list)):
        fname = sample_list[index]
        data = corr_data[fname][0].copy()
        label = corr_data[fname][1]

        idx = 0
        M1 = np.zeros((200,200))
        for j in range(0,200):
            k = 0
            while k < 200:
                if k <= j: k += 1
                else:
                    if idx < data.shape[0]:
                        M1[j][k] = data[idx]
                        idx += 1
                        k += 1
                    else: break

        M2 = M1.transpose()
        eye_M = np.eye(200, dtype=float)
        M3 = M1 + eye_M
        corr_mat.append(np.expand_dims(M1, axis=2))

        onehot = np.zeros((2))
        if label == 1:  onehot[1] = 1
        else : onehot[0] = 1
        onehot_mat.append(onehot)

    return np.array(corr_mat), np.array(onehot_mat)

def gen_mask():
    
    image_size=299
    central_size=200

    idx = 19900
    M1 = np.ones((200,200))
    for j in range(0,200):
        k = 0
        while k < 200:
            if k <= j:
                k += 1
            else:
                if idx < 19900:
                    M1[j][k] = 0
                    idx += 1
                    k += 1
                else: break

    M1 = np.expand_dims(M1, axis=2)
    new_M1  = np.concatenate([M1,M1,M1], axis = -1) 
    new_M1 = np.expand_dims(new_M1, axis=0)

    M = np.pad(new_M1,\
    [[0,0], [int((np.ceil(image_size/2.))-central_size/2.), int((np.floor(image_size/2.))-central_size/2.)],\
    [int((np.ceil(image_size/2.))-central_size/2.), int((np.floor(image_size/2.))-central_size/2.)],\
    [0,0]],'constant', constant_values = 1)

    return M


