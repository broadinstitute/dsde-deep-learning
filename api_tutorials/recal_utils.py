import numpy as np
import numpy.ma as ma
from scipy.stats import entropy


def get_next_batch(generator, quality_correction=0.0):
    batch = next(generator)
    tensor = batch[0][OQ_TENSOR_NAME]
    bqsr = batch[0][BQSR_TENSOR_NAME]
    label = batch[1]
    pred = model.predict_on_batch(tensor)
    pred_qscores = -10 * np.log10(
        pred[:, :, args.labels['BAD_BASE']]) + quality_correction  # +10 only if the tensor is generated with a bias

    orig_qscores = -10 * np.log10(1 - np.max(tensor[:, :, :4], axis=2))
    annot = tensor[:, 0, (args.input_symbols['pair'], args.input_symbols['mq'])]

    return pred_qscores, orig_qscores, bqsr, label, annot


def tensor_to_quality_array(tensor):
    '''
    tensor : (batch_size, 151, 7)

    returns : (batch_size, 151)
    '''
    return -10 * np.log10(1 - np.max(tensor[:, :, :4], axis=2))

def KL_divergence_metric(y_true, y_pred):
    '''	KL divergence metrics for Keras - still under construction '''

    # maybe scikit learn
    predicted_qs = -10*np.log10(y_pred[:,:, args.labels['BAD_BASE']])
    match_qs = (predicted_qs[:,:,np.newaxis] * y_true)[:,:, args.labels['GOOD_BASE']]
    mismatch_qs = (predicted_qs[:,:,np.newaxis] * y_true)[:,:, args.labels['BAD_BASE']]
    match_qs = match_qs[match_qs > 0]
    mismatch_qs = mismatch_qs[mismatch_qs > 0]

    # bins are half open: 1 will go in the [1,2) bin
    max_quality=50
    match_hist, match_bins = np.histogram(np.round(match_qs), bins=max_quality, range = (0,max_quality))
    mismatch_hist, mismatch_bins = np.histogram(np.round(mismatch_qs), bins=max_quality, range = (0,max_quality))

    # compute the KL divergence KL(match||mismatch) - the order chosen arbitrariliy i.e. could've easily chosen KL(mismatch||match)
    # mask bins with 0 probability mass because numpy doens't know 0*log(0)=0
    ma_match_hist = ma.array(match_hist/np.sum(match_hist), mask=match_hist == 0)
    ma_mismatch_hist = ma.array(mismatch_hist/np.sum(mismatch_hist), mask=match_hist == 0)
    print(ma_match_hist)
    print(ma_mismatch_hist)
    print(entropy(ma_match_hist, ma_mismatch_hist))
    KL = -ma.sum(ma_match_hist * ma.log(ma_mismatch_hist)) - ma.sum(- ma_match_hist * ma.log(ma_match_hist))
    return KL

def KL_divergence(match_qs, mismatch_qs):
    ''' compute the KL divergence between the predicted qualities of bases that match the reference and those that don't
    match_qs and mismatch_qs are both arrays of qualities, unsorted and unrounded, straight out of the CNN or SAM.
    greater the KL divergence, the greater the separation between the two distributions
    '''
    # bins are half open: 1 will go in the [1,2) bin
    max_quality=50
    match_hist, match_bins = np.histogram(np.round(match_qs), bins=max_quality, range = (0,max_quality))
    mismatch_hist, mismatch_bins = np.histogram(np.round(mismatch_qs), bins=max_quality, range = (0,max_quality))

    # compute the KL divergence KL(match||mismatch) - the order chosen arbitrariliy i.e. could've easily chosen KL(mismatch||match)
    # mask bins with 0 probability mass because numpy doens't know 0*log(0)=0
    ma_match_hist = ma.array(match_hist, mask=match_hist == 0)
    ma_mismatch_hist = ma.array(mismatch_hist, mask=match_hist == 0)
    KL = -ma.sum(ma_match_hist * ma.log(ma_mismatch_hist)) - ma.sum(- ma_match_hist * ma.log(ma_match_hist))
    return KL