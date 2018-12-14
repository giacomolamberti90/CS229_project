import theano
import numpy
import os
import matplotlib.pyplot as plt
import theano.tensor as tensor
from keras.layers import Embedding,GRU,concatenate
from keras.callbacks import EarlyStopping
import datasource
#from model import VGG_19
from evaluation import t2i, i2t, input2image
from datasets import build_dictionary
from datasets import load_dataset
from theano.tensor.extra_ops import fill_diagonal
from keras.layers import Input, Dense
from keras.layers.core import Lambda, Masking
from keras.models import Model
from keras.utils.vis_utils import plot_model
from hyperopt import Trials, STATUS_OK, tpe,  fmin, tpe, hp, STATUS_OK, Trials
from keras.preprocessing import image
import csv

def compute_errors(s_emb, im_emb):
    """ Given sentence and image embeddings, compute the error matrix """
    erros = [order_violations(x, y) for x in s_emb for y in im_emb]
    return numpy.asarray(erros).reshape((len(s_emb), len(im_emb)))


def order_violations(s, im):
    """ Computes the order violations (Equation 2 in the paper) """
    return numpy.power(numpy.linalg.norm(numpy.maximum(0, s - im)),2)


def l2norm(X):
    """ Compute L2 norm, row-wise """
    norm = tensor.sqrt(tensor.pow(X, 2).sum(1))
    X /= norm[:, None]
    return X


def contrastive_loss(labels, predict):
    """For a minibatch of sentence and image embeddings, compute the pairwise contrastive loss"""
    global model_options
    margin = model_config['margin']
    res = theano.tensor.split(predict, [model_config['output_dim'], model_config['output_dim']], 2, axis=-1)
    s = res[0]
    im = res[1]
    im2 = im.dimshuffle(('x', 0, 1))
    s2 = s.dimshuffle((0, 'x', 1))
    errors = tensor.pow(tensor.maximum(0, im2 - s2), 2).sum(axis=2)
    diagonal = errors.diagonal()
    # compare every diagonal score to scores in its column (all contrastive images for each sentence)
    cost_s = tensor.maximum(0, margin - errors + diagonal)
    # all contrastive sentences for each image
    cost_im = tensor.maximum(0, margin - errors + diagonal.reshape((-1, 1)))
    cost_tot = cost_s + cost_im
    cost_tot = fill_diagonal(cost_tot, 0)
    return cost_tot.sum()


# main trainer
def train(params):

    try:

        #GRID SEARCH
        print (params)

        global  model_config
        model_config['margin'] = params['margin'] if 'margin' in params else model_config['margin']
        model_config['output_dim'] = params['output_dim'] if 'output_dim' in params else model_config['output_dim']
        model_config['max_cap_length'] = params['max_cap_length']  if 'max_cap_length' in params else model_config['max_cap_length']
        model_config['optimizer'] =  params['optimizer']  if 'optimizer' in params else model_config['optimizer'],
        model_config['dim_word'] = params['dim_word'] if 'dim_word' in params else model_config['dim_word']

        # Load training and development sets
        print ('Loading dataset')
        dataset = load_dataset(model_config['data'], cnn=model_config['cnn'])

        train = dataset['train']
        #train['ims'] = train['ims'][0:1000]
        #train['caps'] = train['caps'][0:5000]
        for key, value in train.items():
            print('Size: ' + key + ': ')
            print(len(value))
        test = dataset['test']
        val = dataset['dev']

        # Create dictionary
        print ('Creating dictionary')

        worddict = build_dictionary(train['caps'] + val['caps'])
        print ('Dictionary size: ' + str(len(worddict)))
        model_config['worddict'] = len(worddict)

        embeddings_index = {}
        f = open(model_config['glove.path'])
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = numpy.asarray(values[1:], dtype='float32')
            except:
                print(values[1:])
            embeddings_index[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(embeddings_index))

        embedding_matrix = numpy.zeros((len(worddict) + 2, model_config['dim_word']))
        for word, i in worddict.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i+2] = embedding_vector
            if i < 5:
              print(word)
              print(embedding_matrix[i+2])

        print(embedding_matrix[0])
        print(embedding_matrix[4])

        print ("Image model loading")
        # # this returns a tensor of emb_image
        image_input = Input(shape=(model_config['dim_cnn'],), name='image_input')
        X = Dense(model_config['output_dim'])(image_input)
        X = Lambda(lambda x: l2norm(x))(X)
        emb_image = Lambda(lambda x: abs(x))(X)

        print ("Text model loading")
        # this returns a tensor of emb_cap
        cap_input = Input(shape=(model_config['max_cap_length'],), dtype='int32', name='cap_input')
        X = Masking(mask_value=0,input_shape=(model_config['max_cap_length'], model_config['output_dim']))(cap_input)
        #X = Embedding(output_dim=model_config['dim_word'], input_dim=model_config['worddict']+2, 
        #              input_length=model_config['max_cap_length'])(cap_input)
        X = Embedding(output_dim=model_config['dim_word'], input_dim=len(worddict)+2, input_length=model_config['max_cap_length'], 
                      weights=[embedding_matrix], trainable=True)(cap_input)
        X = GRU(output_dim=model_config['output_dim'], return_sequences=False)(X)
        X = Lambda(lambda x: l2norm(x))(X)
        emb_cap = Lambda(lambda x: abs(x))(X) 

        print ("loading the joined model")
        # merged = _Merge( mode='concat')([emb_cap, emb_image])
        merged = concatenate([emb_cap, emb_image])
        model = Model(inputs=[cap_input, image_input], outputs=[merged])

        print ("compiling the model")
        model.compile(optimizer='adam', loss=contrastive_loss)

        # load weights
        model.load_weights('my_model_weights_71.h5')

        def eval_model():
            
            print ('evaluating model...')
            weights = model.get_weights()

            # weights
            emb_w = weights[0]
            im_w = weights[4]
            im_b = weights[5]
            gru_weights = weights[1:4]
           
            # image model
            test_model_im = Model(inputs=image_input, outputs=emb_image)
            test_model_im.set_weights([im_w, im_b])
            test_model_im.compile(optimizer='adam', loss=contrastive_loss)
            
            test_iter = datasource.Datasource(test, worddict=worddict)         
            _, test_ims = test_iter.all()

            # predicted images
            pred_ims = test_model_im.predict(test_ims)
           
            # caption model
            test_model_cap = Model(inputs=cap_input, outputs=emb_cap)
            test_model_cap.set_weights([emb_w]+gru_weights)
            test_model_cap.compile(optimizer='adam', loss=contrastive_loss)            
            
            caps = []
            #input_cap = test['caps'][100]
            input_cap = input("Insert caption: ").encode('ascii')
            caps.append(input_cap.strip())
            print(input_cap)
            
            test_input = {}
            test_input['ims']  = []
            test_input['caps'] = caps
           
            test_iter = datasource.Datasource(test_input, batch_size=1, worddict=worddict)    
            test_cap, _ = test_iter.all()
            
            # predicted caption
            pred_cap = test_model_cap.predict(test_cap)

            # compute error matrix
            test_errs = compute_errors(pred_cap, pred_ims)
            
            # indices of 10 most likely pictures in test set
            ind_ims = input2image(test_errs)
            print(ind_ims)
                
            for i in ind_ims[0]:
                print(val['caps'][5*i])
    
            directory = '../data/coco/'
            #imgs = sorted(os.listdir(directory))
               
            with open('../data/coco/test_path.txt', 'r') as f:
                imgs = f.readlines() 
            
            for i in ind_ims[0]:
               
               img_path = directory + imgs[i][1:-2]              
               print('Image: ', os.fsdecode(imgs[i]))
               
               img = image.load_img(img_path, target_size=(224, 224)) 
               
               plt.imshow(img)
               plt.show()
            
        #evaluate model
        eval_model()

    except:
        raise

def trainer(config):
    
    global model_config
    model_config = config
    train(model_config)

