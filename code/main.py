'''
Author: Igor Lapshun
'''

#This is the configurations/ meta parameters for this model
# (currently set to optimal upon cross valiation).
config = {
    'model_cnn':'/home/lucas/cs229/image_caption/data/vgg19_weights.h5',
    'data': '/home/lucas/cs229/image_caption/data/coco',
    'save_dir': 'anypath',
    'dim_cnn': 4096,
    'optimizer': 'adam',
    'batch_size': 128,
    'epoch': 50,
    'output_dim': 1024,
    'dim_word': 300,
    'lrate': 0.05,
    #'lrate': 0.0005,
    'max_cap_length' : 50,
    'cnn' : '10crop',
    'margin': 0.05,
    'attn_rnn_dim': 100, 
    'glove.path': '/home/lucas/cs229/image_caption/data/glove.6B.300d.txt'
}


if __name__ == '__main__':
    #import eval_only
    #eval_only.trainer(config)
    #import trainer_bidirection
    #trainer_bidirection.trainer(config)
    import trainer
    trainer.trainer(config)
