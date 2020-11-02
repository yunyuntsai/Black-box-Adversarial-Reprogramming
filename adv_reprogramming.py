import random
import os
import numpy as np
from time import time
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import pickle
from scipy import stats
import math
import csv
import utils
import cal_func, mapping_func, estimator
from nets import resnet_v2, inception_v3, inception




class Adversarial_Reprogramming():

    def __init__(self):

        self.network_name = 'inception_v3'
        self.sample_dir = './inception_sample'
        self.train_dir = './inception_train'
        self.pretrained_model_dir = './model'
        self.batchsize = 10
        self.max_epoch = 9
        self.image_size = 299
        self.central_size = 200
        self.num_rand_vec = 1
        self.q_batch = 1
        self.lr = 1e-1
        self.lamda = 10
        self.save_freq = 1
        self.mlm_num = 6


    def adv_program(self, central_image, isTraining):
        
        #generate adversarial samples with adversarial program function
        if self.image_size == 299:
            means = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32) 
        elif self.image_size == 224:
            means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        self.M = tf.constant(utils.gen_mask(), dtype=tf.float32)
        
        with tf.variable_scope('adv_program',reuse=tf.AUTO_REUSE):
            self.W = tf.get_variable('program',shape=[1,self.image_size,self.image_size,3], dtype = tf.float32)
        
        self.beta = 0.1
        
        var_size = self.image_size*self.image_size*3
        var_noises = tf.random_normal(mean=0, stddev=1000, dtype=tf.float32, shape=(self.q_batch, var_size))
        var_noises = tf.math.l2_normalize(var_noises, axis=1)
        var = tf.concat((self.W, self.W + self.beta*tf.reshape(var_noises, shape=[self.q_batch, self.image_size, self.image_size,3])), axis=0) #todo
  
        central_image  = tf.concat([central_image, central_image, central_image], axis = -1) 
        
        self.X = tf.pad(central_image, paddings = tf.constant([[0,0], [int((np.ceil(self.image_size/2.))-self.central_size/2.),\
                         int((np.floor(self.image_size/2.))-self.central_size/2.)],\
                         [int((np.ceil(self.image_size/2.))-self.central_size/2.),\
                          int((np.floor(self.image_size/2.))-self.central_size/2.)], [0,0]]))
        
        all_X_adv = []
        for i in range(self.q_batch+1):
            self.P = tf.nn.tanh(tf.multiply(var[i], self.M))

            X_adv =  self.P + self.X

            self.channels = tf.split(X_adv, axis=3, num_or_size_splits=3)
            for i in range(3):
                self.channels[i] -= means[i]
                self.channels[i] /= std[i]

            all_X_adv.append(X_adv)
        all = []
        all = tf.concat([all_X_adv[0]], 0)
        for j in range(len(all_X_adv)-1):
            all = tf.concat([all, all_X_adv[j+1]], 0)
        return all, var_noises  


    def run(self, flist, labels):

        kf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        all_corr = pickle.load(open('./correlations_file'+p_ROI+'.pkl', 'rb'))
        np.random.shuffle(flist)
        y_arr = np.array([utils.get_label(f, labels) for f in flist])


        input_images  = tf.placeholder(shape = [None,self.central_size,self.central_size,1], dtype = tf.float32)
        Y = tf.placeholder(tf.float32, shape=[None, 2]) 
        MLM_Index = tf.placeholder(tf.int32, shape=[None,])

        isTraining = tf.placeholder(tf.bool)
        train_mode = tf.count_nonzero(isTraining)

        if self.network_name == 'resnet_v2_50':
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                
                adv_data, var_noises = self.adv_program(input_images, isTraining)
                self.imagenet_logits, self.imagenet_prob = resnet_v2.resnet_v2_50(adv_data, num_classes = 1001,is_training=False)
                
                self.imagenet_prob = self.imagenet_prob[:,1:]
                self.top_k_value, self.top_k_indices = tf.math.top_k(self.imagenet_prob, 30)
                self.freq_prob = mapping_func.freq_mapping(self.imagenet_prob, MLM_Index)
                
                #self.disturbed_prob = tf.matmul(self.imagenet_prob, mapping_func.label_mapping())
                #self.multi_label_prob = mapping_func.multi_prob(self.disturbed_prob)
                
                init_fn = slim.assign_from_checkpoint_fn(os.path.join(self.pretrained_model_dir,self.network_name+'.ckpt'),slim.get_model_variables('resnet_v2_50'))

        if self.network_name == 'inception_v3':
            print("using inception")
            with slim.arg_scope(inception.inception_v3_arg_scope()):

                adv_data, var_noises = self.adv_program(input_images, isTraining)

                print('adversarial image shape:', adv_data.shape)
                
                self.imagenet_logits, self.imagenet_prob, self.pre_logits = inception_v3.inception_v3(adv_data, num_classes = 1001,is_training=False)
                
                
                self.imagenet_prob = self.imagenet_prob[:,1:]
                self.top_k_value, self.top_k_indices = tf.math.top_k(self.imagenet_prob, 30)
                self.freq_prob = mapping_func.freq_mapping(self.imagenet_prob, MLM_Index)
                
                #self.disturbed_prob = tf.matmul(self.imagenet_prob, self.label_mapping())  
                #self.multi_label_prob = self.multi_prob(self.disturbed_prob)
                
                init_fn = slim.assign_from_checkpoint_fn(os.path.join(self.pretrained_model_dir,self.network_name+'.ckpt'), slim.get_model_variables('InceptionV3'))


        ## Compute Gradient
        y1 = tf.tile(Y, [self.q_batch+1, 1])
        #self.cross_entropy_loss =   -tf.reduce_sum(y1 *tf.log(self.freq_prob + 1e-6), axis=1)
        self.focal_loss = cal_func.multi_focal_loss(4, 0.8, y1, self.freq_prob)
        estimate_grad = estimator.func(self.focal_loss, self.q_batch, var_noises) 
        eGrad = tf.placeholder(tf.float32, shape=[1, self.image_size, self.image_size,3]) 
        gradt = tf.gradients(self.focal_loss[0:self.batchsize], self.W)[0]   

        ## Optimize
        global_steps =tf.Variable(0, trainable=False)
        starter_learning_rate = 0.01
        end_learning_rate = 0.001
        decay_steps = 2000
        decay_rate = 0.96
        learning_rate = tf.train.polynomial_decay(starter_learning_rate,global_steps,decay_steps, end_learning_rate, power=0.9)
        gradvars = [(gradt , self.W)]
        optim =  tf.train.AdamOptimizer(0.01)
        Step = optim.apply_gradients(gradvars, global_step= global_steps)

        ## Compute accuracy
        correct_prediction = tf.equal(tf.argmax(self.freq_prob[0:self.batchsize],1), tf.argmax(Y,1))
        predict_probilities = tf.reduce_sum(self.freq_prob[0:self.batchsize] * Y, axis=-1)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        ## Training
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)       

        ##Start training with mini batch size
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        init_opt = tf.global_variables_initializer()
        
        ## run cross-validation trainining with 10 split training set
        for kk,(train_index, test_index) in enumerate(kf.split(flist, y_arr)):
            
            sess.run(init_opt)
            init_fn(sess)    

            train_samples = np.array(flist)[train_index]
            test_samples = np.array(flist)[test_index]
            train_data, train_label = utils.process_data(all_corr, train_samples)
            test_data, test_label = utils.process_data(all_corr, test_samples)

            total_batch = int(train_data.shape[0]/self.batchsize)

            epoch_starttime = time()                
            


            Asd = np.zeros((1000,2))
            NonAsd = np.zeros((1000,2))
            asd_cnt = 0
            nonasd_cnt = 0
            for i in range(train_data.shape[0]): 
                x_train_batch = np.expand_dims(train_data[i], axis=0)
                y_train_batch = np.expand_dims(train_label[i], axis=0)

                adv_image_batch, Top_k_indices, Top_k_values = sess.run([adv_data, self.top_k_indices, self.top_k_value], feed_dict={input_images: x_train_batch, Y: y_train_batch, isTraining:True})

                if y_train_batch[0][0] == 1:
                    for i in range(30):
                        NonAsd[Top_k_indices[0][i]][0] += Top_k_values[0][i]
                        NonAsd[Top_k_indices[0][i]][1] += 1
                        asd_cnt += 1

                elif y_train_batch[0][1] == 1:
                    for i in range(30):
                        Asd[Top_k_indices[0][i]][0] += Top_k_values[0][i]
                        Asd[Top_k_indices[0][i]][1] += 1
                        nonasd_cnt += 1
            mlm_index = mapping_func.freq_idx_search(NonAsd, Asd, self.mlm_num)
            
           
            best_result = np.zeros(4)
            for epoch in range(self.max_epoch): 
                epoch_starttime = time()
                batch_loss = []
                total_loss = []

                #np.append(topK_array, Top_k_indices[0:self.batchsize], axis=0)
                #shuffle#
                s = np.arange(train_data.shape[0])
                s = np.random.shuffle(s)
                train_data = train_data[s][0]
                train_label = train_label[s][0]

                for batch in range(total_batch): 
                    loss_list = []
                    est_Glist = []                   
                    x_train_batch = train_data[batch*self.batchsize:(batch+1)*self.batchsize]
                    y_train_batch = train_label[batch*self.batchsize:(batch+1)*self.batchsize]
                    for q_batch in range(int(self.num_rand_vec/self.q_batch)): 
                        adv_image_batch, Losses, est_Grad,  tr_mode= sess.run([adv_data, self.focal_loss, estimate_grad, train_mode], feed_dict={input_images: x_train_batch, Y: y_train_batch, MLM_Index: mlm_index, isTraining:True})
                        loss_list.append(np.mean(Losses, axis=0))
                        est_Glist.append(est_Grad)

                    avg_loss = np.mean(loss_list, axis=0, keepdims=False)
                    batch_loss.append(avg_loss)
                    avg_Glist = np.mean(est_Glist, axis=0, keepdims = True)
                        
                    #weight, _ = sess.run([self.W, Step], feed_dict = {eGrad: avg_Glist, MLM_Index:mlm_index, isTraining:True}) 
                    weight, _ = sess.run([self.W, Step], feed_dict = {input_images:  x_train_batch, Y: y_train_batch, MLM_Index:mlm_index, isTraining:True}) 

                    ## validate performance per 5 mini batch 
                    if batch % 10 == 0:                    
                        valid_batch_acc, pred_probs, img_X_adv, tr_mode, LR, gp = sess.run([accuracy, predict_probilities, adv_data,  train_mode, learning_rate,  global_steps], \
                                                                feed_dict = {input_images:  x_train_batch,Y: y_train_batch, MLM_Index:mlm_index, isTraining:False})                    
 
                        valid_acc = float(valid_batch_acc/self.batchsize)
                        #print('train mode: ', tr_mode)
                        print('epoch:{:03d}/{:03d}, batch: {:04d}/{}, loss: {:.4f}, valid_acc: {:.2f}'.format(epoch,\
                                                                self.max_epoch,batch,total_batch ,np.mean(batch_loss), valid_batch_acc))
                        total_loss.append(np.mean(batch_loss))
                        batch_loss = []

                ## save model per epoch
                if (epoch+1) % self.save_freq == 0:
                    saver.save(sess, os.path.join(self.train_dir+'_'+str(kk), 'model_{:06d}.ckpt'.format(epoch+1)))
                    print('model_{:06d}.ckpt saved'.format(epoch+1)) 
                
                ## End of Training
                epoch_duration = time()- epoch_starttime
                print("Training this epoch takes:","{:.2f}".format(epoch_duration))
                print("Total average loss is:","{:.2f}".format(np.mean(total_loss)))

               
                testing_start = time()    
                test_total_batch = int(test_data.shape[0]/self.batchsize)
                Test_acc = 0.0
                test_acc_sum = 0.0
                test_sen_sum = 0.0
                test_spe_sum = 0.0
                
                for i in range(test_total_batch):
                    test_image_batch = test_data[i*self.batchsize:(i+1)*self.batchsize]
                    test_label_batch = test_label[i*self.batchsize:(i+1)*self.batchsize]

                    test_acc, test_batch_result, tr_mode = sess.run([accuracy,self.freq_prob[0:self.batchsize], train_mode], feed_dict = {input_images:test_image_batch,Y:test_label_batch, MLM_Index:mlm_index,isTraining:False})
                    

                    test_batch_acc, test_batch_sen, test_batch_spe = cal_func.confusion(np.argmax(test_batch_result, axis=1), np.argmax(test_label_batch, axis=1))
                    label = np.argmax(test_label_batch[0], axis=0)

                    Test_acc += test_acc
                    
                    test_acc_sum += test_batch_acc
                    test_sen_sum += test_batch_sen
                    test_spe_sum += test_batch_spe

                final_acc = float(Test_acc/test_total_batch)
                test_acc = float(test_acc_sum/test_total_batch)
                test_sen = float(test_sen_sum/test_total_batch)
                test_spe = float(test_spe_sum/test_total_batch)
                testing_duration = time()-testing_start
                if test_acc > best_result[1]: 
                    best_result[:] = [epoch, test_acc, test_sen, test_spe]
                elif test_acc == best_result[1]:
                    if test_sen > best_result[2]:
                        best_result[:] = [epoch, test_acc, test_sen, test_spe]
                    elif test_sen == best_result[2]:
                        if test_spe > best_result[3]:
                            best_result[:] = [epoch, test_acc, test_sen, test_spe]
                #print('Test accuracy: {:.4f}'.format(final_acc))
                #print('test accuracy: {:.4f} | test sensitivity: {:.4f} | test speciality: {:.4f}'.format(test_acc,test_sen, test_spe)) 
            print('best accuracy: {:.4f} | best sensitivity: {:.4f} | best speciality: {:.4f}'.format(best_result[1],best_result[2], best_result[3]))
            with open('./check/ASD_withMLM_accuracy_focal.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow([kk, best_result[0], best_result[1], best_result[2], best_result[3]])  
            print("Testing finished takes:{:.2f} secs".format(testing_duration)) 

            

if __name__ == "__main__":

    p_ROI = 'cc200'
    data_main_path = '/home/yunyun/acerta-abide/data/functionals/cpac/filt_global/rois_'+p_ROI #cc200'#path to time series data
    flist = os.listdir(data_main_path)
    print('flist: ', len(flist))    
    
    for f in range(len(flist)):
        flist[f] = utils.get_key(flist[f])
    df_labels = pd.read_csv('/home/yunyun/acerta-abide/data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv') #path 
    df_labels.DX_GROUP = df_labels.DX_GROUP.map({1: 1, 2:0})  

    labels = {}
    for row in df_labels.iterrows():
        file_id = row[1]['FILE_ID']
        y_label = row[1]['DX_GROUP']
        if file_id == 'no_filename':
            continue
        assert(file_id not in labels)
        labels[file_id] = y_label

    model = Adversarial_Reprogramming()
    model.run(flist, labels)
