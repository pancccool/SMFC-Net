import os
import scipy.io as sio
import numpy as np
from sklearn.utils import shuffle


def sym_wsr(data,n_subjects=125, n_regions=120, n_lambda=11):
    '''
        Symmetrization of WSR.
    '''
    _sym_data = np.zeros((n_lambda,n_subjects,n_regions,n_regions))
    for param_lambda1 in range(n_lambda):
        tmpFeature = data[param_lambda1]
        sym_Net = np.zeros((n_subjects,n_regions,n_regions))
        for i in range(n_subjects):
            originalNet = tmpFeature[i]
            originalNet = (originalNet + originalNet.T) / 2
            sym_Net[i,:] = originalNet
        _sym_data[param_lambda1, :] = sym_Net
    return _sym_data


nor_num =
pat_num =
def load_data(dataFile,n_subjects, n_regions, n_lambda, nor_num = nor_num, pat_num = pat_num):
    '''
    :param dataFile: location of Sparsity brain network.
    :param n_subjects:  all subjects numbers
    :param n_regions:  all brain regions
    :param n_lambda:   Sparsity parameter
    :return:  all data
    '''
    data = sio.loadmat(dataFile)
    BrainNetSet = data["BrainNetSet"]
    tupleBrainNetSet = ()
    for i in range(n_lambda):
        a = np.array(list(BrainNetSet[i])).reshape(n_subjects, n_regions, n_regions)
        tupleBrainNetSet += (a,)
    nor = np.zeros((nor_num,),dtype = int)
    pat = np.ones((pat_num,),dtype = int)
    label = np.concatenate((nor,pat))
    arrayBrainNetSet = np.array((tupleBrainNetSet))

    if dataFile == r'data\BrainNetSet_SZ_WSR.mat':
        arrayBrainNetSet = np.abs(sym_wsr(arrayBrainNetSet))  

    return arrayBrainNetSet,label

def threshold(data,n_subjects, n_regions, n_lambda):
    '''
    Binarization of wsr.
    '''
    threshold_final = np.zeros((n_lambda, n_subjects, n_regions, n_regions))
    for i in range(n_lambda):
        threshold = np.zeros((n_subjects, n_regions, n_regions))
        for j in range(n_subjects):
            sub_wsr = data[i][j]
            _single_threshold = (sub_wsr != 0).astype(int)
            threshold[j,:] = _single_threshold
        threshold_final[i,:] = threshold
    return threshold_final


def sparse_guided(data_wsr_threshold,data_single_PC, n_subjects, n_regions, n_lambda):
    '''
    PC is sparse-guided by binarized WSR
    '''
    threshold_for_PC = np.zeros((n_lambda, n_subjects, n_regions, n_regions))
    for i in range(n_lambda):
        _PC_ = np.zeros((n_subjects, n_regions, n_regions))
        for j in range(n_subjects):
            s_threshold = data_wsr_threshold[i][j]
            s_pc = data_single_PC[j]
            _wsr_thred_PC = s_threshold * s_pc
            _PC_[j, :] = _wsr_thred_PC
        threshold_for_PC[i, :] = _PC_
    return threshold_for_PC

def transform(sparse_guided_matrix,n_subjects, n_regions, n_lambda):
    '''
    :return: transformed matrix
    '''
    threshold_ten = sparse_guided_matrix[1:, :]
    threshold_change = np.zeros((n_subjects,n_lambda-1, n_regions, n_regions))
    for ii in range(n_subjects):
        b = threshold_ten[:, ii, :]
        threshold_change[ii, :] = b
    return threshold_change

def load_SMFCN_label(n_subjects = 125,n_regions = 120, n_lambda = 11):
    dataFile_PC = r'data\BrainNetSet_HC_SZ_PC.mat'
    dataFile_WSR = r'data\BrainNetSet_SZ_WSR.mat'
    pc, label = load_data(dataFile_PC,n_subjects,n_regions,n_lambda)
    wsr, label = load_data(dataFile_WSR,n_subjects,n_regions,n_lambda)
    _single_pc = pc[0]
    binarization_wsr = threshold(wsr,n_subjects,n_regions,n_lambda)
    _sparse_guided = sparse_guided(binarization_wsr,_single_pc,n_subjects,n_regions,n_lambda)
    sparse_guided_matrix = transform(_sparse_guided,n_subjects,n_regions,n_lambda)
    data_all,label_all = shuffle(sparse_guided_matrix,label,random_state=0)
    return data_all,label_all