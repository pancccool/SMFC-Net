import os
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import load_model
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from our.open_code.load_data import load_SMFCN_label
from our.open_code.SMFCNet import SMFC_Net
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


ex_num = 10
ex_fold_num = 5
in_num = 5
in_fold_num = 5
classifiers = in_num * in_fold_num
learning_rate = 1e-3
batch_size = 20
epochs = 200
test_num = 25
SBN_num = 10
data_all,label_all = load_SMFCN_label()



all_mean_result_ex = np.array(())
all_result_ex = np.zeros((ex_num, ex_fold_num))
for jj in list(range(ex_num)):
    np.random.seed(jj)
    random.seed(jj) 
    tf.random.set_seed(jj)
    kf = StratifiedKFold(n_splits=ex_fold_num,shuffle=True,random_state=jj)
    h = 0
    everyresult_ex = np.array(())
    for train_index, test_index in kf.split(data_all,label_all):
        h +=1
        train_X, train_y = data_all[train_index], label_all[train_index]
        test_X, test_y = data_all[test_index], label_all[test_index]
        num_train_data_ex = train_X.shape[0]
        train_X = np.expand_dims(train_X,axis=-1)
        test_X = np.expand_dims(test_X,axis=-1)

        all_mean_result = np.array(())
        in_all_result = np.zeros((in_num,in_fold_num))
        for pp in range(in_num):
            kf_in = StratifiedKFold(n_splits=in_fold_num,shuffle=True,random_state=pp)
            everyresult = np.array(())
            m = 0
            for train_index_in,dev_index in kf_in.split(train_X,train_y):  
                m +=1
                train_X_in,train_y_in = train_X[train_index_in],train_y[train_index_in]
                dev_X,dev_y = train_X[dev_index],train_y[dev_index]
                num_train_data_in = train_X_in.shape[0]
                
                model = SMFC_Net()
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                              loss=tf.keras.losses.sparse_categorical_crossentropy,
                              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
                data_list = [] 
                for i in range(SBN_num):
                    a = train_X_in[:,i,:]
                    data_list.append(a)
                inputs_list = [] 
                for i in range(SBN_num):
                    a = dev_X[:,i, :]
                    inputs_list.append(a)

                model.fit(x=data_list, y=train_y_in, batch_size=batch_size, epochs=epochs,verbose=0)

                _, train_acc = model.evaluate(data_list, train_y_in, verbose=0)
                _, dev_acc = model.evaluate(inputs_list, dev_y, verbose=0)
                print('Train: %.3f, Test: %.3f' % (train_acc, dev_acc))
                model.save(r'trained_model\model_exnum_%d_exfold_%d_innum_%d_infold_%d.h5' % (jj,h,pp,m))
                everyresult = np.append(everyresult, model.evaluate(inputs_list,dev_y)[1])
            in_all_result[pp,:] = everyresult 
            mean_acc_in = (np.sum(everyresult)) / in_fold_num
            all_mean_result = np.append(all_mean_result,mean_acc_in)
        acc_final = (np.sum(all_mean_result)) / in_num 

        inputs_list_test = [] 
        for i in range(10):
            a = test_X[:, i, :]
            inputs_list_test.append(a)

        all_ex_result = ()
        # The best twenty-five classifiers make predictions on the test set separately and perform majority voting.
        for innum in range(in_num):
            every_fold_ex = np.zeros((in_fold_num,test_num))
            for infold in range(in_fold_num):
                nbmodel= load_model(r'trained_model\\model_exnum_%d_exfold_%d_innum_%d_infold_%d.h5' % (jj,h,innum,infold+1),compile=False)

                y_pred = nbmodel(inputs_list_test, training=False).numpy()
                y_predddd = np.argmax(y_pred, axis=-1)
                every_fold_ex[infold,:] = y_predddd
            all_ex_result += (every_fold_ex,)
        all_ex_result_y_pred = np.vstack((all_ex_result[0],all_ex_result[1],all_ex_result[2],all_ex_result[3],all_ex_result[4]))

        y_pred_fusion = np.sum(all_ex_result_y_pred,axis=0)
        y_pred_final = np.array(())
        y_fusion_prob = np.zeros((test_num,2))
        for bb in range(test_num):
            one_prob = y_pred_fusion[bb] /classifiers
            zero_prob = 1-(y_pred_fusion[bb] /classifiers)
            prob = np.array((zero_prob,one_prob))
            y_fusion_prob[bb,:] = prob

            if (y_pred_fusion[bb] / classifiers)>=0.5:
                y_pred_final = np.append(y_pred_final,1)
            else:
                y_pred_final = np.append(y_pred_final,0)
        y_true_ex = test_y


        final_acc = sum(y_pred_final == y_true_ex) / test_num
        print("each_fold_acc_ex: {:.7f}".format(final_acc))
        everyresult_ex = np.append(everyresult_ex, final_acc)


    all_result_ex[jj,:] = everyresult_ex
    mean_acc_ex = (np.sum(everyresult_ex)) / ex_fold_num
    all_mean_result_ex = np.append(all_mean_result_ex,mean_acc_ex)
    print("mean_acc_ex: {:.7f}".format(mean_acc_ex))

acc_final_ex = (np.sum(all_mean_result_ex)) / ex_num
print("acc_final_ex: {:.7f}".format(acc_final_ex))


