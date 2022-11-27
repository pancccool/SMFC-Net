import os
import numpy as np
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import tensorflow as tf
import random
from tensorflow.keras.models import load_model
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import pandas as pd
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

data_all,label_all = load_SMFCN_label()



all_mean_result_ex = np.array(())
all_mean_result_ex_SEN = np.array(())
all_mean_result_ex_SPE =np.array(())
all_mean_result_ex_AUC = np.array(())
all_mean_result_ex_F1 = np.array(())

all_result_ex = np.zeros((ex_num, ex_fold_num))
all_result_ex_SEN = np.zeros((ex_num, ex_fold_num))
all_result_ex_SPE= np.zeros((ex_num, ex_fold_num))
all_result_ex_AUC = np.zeros((ex_num, ex_fold_num))
all_result_ex_F1 = np.zeros((ex_num, ex_fold_num))

seed = [9,8,7,6,5,4,3,2,1,0] 
for jj in seed: 
    np.random.seed(jj)
    random.seed(jj) 
    tf.random.set_seed(jj)
    kf = StratifiedKFold(n_splits=ex_fold_num,shuffle=True,random_state=jj)
    h = 0
    everyresult_ex = np.array(())
    everyresult_ex_SEN = np.array(())
    everyresult_ex_SPE = np.array(())
    everyresult_ex_AUC = np.array(())
    everyresult_ex_F1 = np.array(())
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
            for train_index_in,dev_index in kf_in.split(train_X,train_y):  # 内序5折
                m +=1
                train_X_in,train_y_in = train_X[train_index_in],train_y[train_index_in]
                dev_X,dev_y = train_X[dev_index],train_y[dev_index]
                num_train_data_in = train_X_in.shape[0]
                
                model = SMFC_Net()
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                              loss=tf.keras.losses.sparse_categorical_crossentropy,
                              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
                data_list = [] 
                for i in range(10):
                    a = train_X_in[:,i,:]
                    data_list.append(a)
                inputs_list = [] 
                for i in range(10):
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

        np.savetxt(r"result\indicator_in\exnum_%d_exfold_%d_of_in_acc.txt" %
                   (jj,h), acc_final.reshape(-1,1) ,fmt="%.18f",delimiter=',')


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
        SEN_TPR =metrics.recall_score(y_true_ex,y_pred_final,pos_label=1)
        SPE_TNR = metrics.recall_score(y_true_ex,y_pred_final,pos_label=0)
        AUC = metrics.roc_auc_score(y_true_ex,y_fusion_prob[:, 1])
        F1 = metrics.f1_score(y_true_ex,y_pred_final)

        everyresult_ex = np.append(everyresult_ex, final_acc)
        everyresult_ex_SEN = np.append(everyresult_ex_SEN, SEN_TPR)
        everyresult_ex_SPE = np.append(everyresult_ex_SPE, SPE_TNR)
        everyresult_ex_AUC = np.append(everyresult_ex_AUC, AUC)
        everyresult_ex_F1 = np.append(everyresult_ex_F1, F1)

    all_result_ex[jj,:] = everyresult_ex
    all_result_ex_SEN[jj, :] = everyresult_ex_SEN
    all_result_ex_SPE[jj, :] = everyresult_ex_SPE
    all_result_ex_AUC[jj, :] = everyresult_ex_AUC
    all_result_ex_F1[jj, :] = everyresult_ex_F1

    mean_acc_ex = (np.sum(everyresult_ex)) / ex_fold_num
    mean_sen_ex = (np.sum(everyresult_ex_SEN)) / ex_fold_num
    mean_spe_ex = (np.sum(everyresult_ex_SPE)) / ex_fold_num
    mean_auc_ex = (np.sum(everyresult_ex_AUC)) / ex_fold_num
    mean_f1_ex = (np.sum(everyresult_ex_F1)) / ex_fold_num

    all_mean_result_ex = np.append(all_mean_result_ex,mean_acc_ex)
    all_mean_result_ex_SEN = np.append(all_mean_result_ex_SEN,mean_sen_ex)
    all_mean_result_ex_SPE = np.append(all_mean_result_ex_SPE,mean_spe_ex)
    all_mean_result_ex_AUC = np.append(all_mean_result_ex_AUC,mean_auc_ex)
    all_mean_result_ex_F1 = np.append(all_mean_result_ex_F1,mean_f1_ex)

    mean_ex_dataframe = pd.DataFrame(
        list({
            'acc':mean_acc_ex,
            'sen':mean_sen_ex,
            'spe':mean_spe_ex,
            'auc':mean_auc_ex,
            'f1':mean_f1_ex,
        }.items()),
    )
    outputpath1 = r'result\indicator_ex\every_mean_acc_num_%d.csv'% (jj)
    mean_ex_dataframe.to_csv(outputpath1, sep=',', index=True, header=True)

    all_ex_dataframe = pd.DataFrame(
        {
            'n-fold':['1','2','3','4','5'],
            'acc':list(everyresult_ex),
            'sen':list(everyresult_ex_SEN),
            'spe':list(everyresult_ex_SPE),
            'auc':list(everyresult_ex_AUC),
            'f1':list(everyresult_ex_F1),
        },
    )
    outputpath2 = r"result\indicator_ex\all_acc_num_%d.csv" % (jj)
    all_ex_dataframe.to_csv(outputpath2, sep=',', index=True, header=True)


acc_final_ex = (np.sum(all_mean_result_ex)) / ex_num
sen_final_ex = (np.sum(all_mean_result_ex_SEN)) / ex_num
spe_final_ex = (np.sum(all_mean_result_ex_SPE)) / ex_num
auc_final_ex = (np.sum(all_mean_result_ex_AUC)) / ex_num
f1_final_ex = (np.sum(all_mean_result_ex_F1)) / ex_num

final_ex = pd.DataFrame(
    list({
        'acc': acc_final_ex,
        'sen': sen_final_ex,
        'spe': spe_final_ex,
        'auc': auc_final_ex,
        'f1': f1_final_ex,
    }.items())
)
outputpath3 = r"result\indicator_ex\final_ex.csv"
final_ex.to_csv(outputpath3, sep=',', index=True, header=True)

