import tensorflow as tf



def func(train_loss, iNum, var_noises):


    image_size = 299
    batchsize = 10
    q_batch =  1


    losses = []
    glist = []

    ##Set paramters    
    beta = 0.1
    d = image_size*image_size*3 
    b_constant = d

    ##gradient-free optimization 
    for i in range(iNum+1):
        loss = train_loss[i* batchsize : i*batchsize + batchsize]
        losses.append(loss)
            
    for i in range(0,len(losses)-1):
        v = tf.expand_dims(var_noises[i], axis=0)
        l = tf.expand_dims(losses[i+1] - losses[0], 1)

        mul = tf.matmul(l,v)
        grad  = b_constant * mul / beta         
        glist.append(grad)

    glist = tf.stack(glist, axis=0)
    print("glist: ",glist.shape)
    avg_grad = tf.reduce_sum(glist, 0) / q_batch
    print(avg_grad.shape)
    estimate_grad = tf.reduce_sum(avg_grad, axis=0) / batchsize
    return tf.reshape(estimate_grad , shape=[image_size, image_size, 3])