"""
Evaluation code for multimodal ranking
Throughout, we assume 5 captions per image, and that
captions[5i:5i+5] are GT descriptions of images[i]
"""
import numpy


import datasets
from datasource import Datasource


def input2image(c2i):
    """
    Input caption->Images (Image Search)
    c2i: (1, N) vector of caption to image errors
    returns indices of 10 most likely pictures
    """
    
    inds = numpy.zeros(c2i.shape, dtype='int32')
    
    for i in range(c2i.shape[0]):
        inds[i,:] = numpy.argsort(c2i[i,:])
    
    return inds[:,0:10]

def t2i(c2i):
    """
    Text->Images (Image Search)
    c2i: (5N, N) matrix of caption to image errors
    vis_details: if true, return a dictionary for ROC visualization purposes
    """

    ranks = numpy.zeros(c2i.shape[0])

    for i in range(len(ranks)):
        d_i = c2i[i]
        inds = numpy.argsort(d_i)

        rank = numpy.where(inds == int(i/5))[0][0]
        ranks[i] = rank

        def image_dict(k):
            return {'id': k, 'score': float(d_i[k])}

    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    meanr = ranks.mean() + 1

    stats = map(float, [r10, meanr])

    return stats



def i2t(c2i):
    """
    Text->Images (Image Search)
    c2i: (5N, N) matrix of caption to image errors
    """

    ranks = numpy.zeros(c2i.shape[1])

    for i in range(len(ranks)):
        d_i = c2i[:, i]
        inds = numpy.argsort(d_i)

        rank = numpy.where((inds/5).astype(int) == i)[0][0]
        ranks[i] = rank

    # Compute metrics
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    meanr = ranks.mean() + 1
    return map(float, [r10, meanr])
