
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import configparser
import argparse
import numpy as np
import tfds_sketchy

#---------------------------------------------------------------------------------------
def map_func(example_serialized):
    image_sketch = example_serialized['sketch']
    image_photo = example_serialized['photo']

    # Resize images
    image_sketch = tf.image.resize(image_sketch, [256, 256])
    image_photo = tf.image.resize(image_photo, [256, 256])

    # Random crop
    image_sketch = tf.image.random_crop(image_sketch, size=[224, 224, 3])
    image_photo = tf.image.random_crop(image_photo, size=[224, 224, 3])

    # Cast to float32
    image_sketch = tf.cast(image_sketch, tf.float32)
    image_photo = tf.cast(image_photo, tf.float32)

    return image_sketch, image_photo

        
AUTO = tf.data.AUTOTUNE
#---------------------------------------------------------------------------------------
import models.siamese as siamese

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type = str, required = True)    
    parser.add_argument('-model', type = str, choices = ['RESNET'], required = True)
    parser.add_argument('-gpu', type = int, required = False) # gpu = -1 set for using all gpus
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read(args.config)
    model_name = args.model
    config_model = config[model_name]
    config_data = config['DATA']
    dataset_name = config_data.get('DATASET')
    
    gpu_id = 0
    if not args.gpu is None :
        gpu_id = args.gpu
        
    ds = tfds.load('tfds_sketchy')        
    ds_train = ds['train']    
    ds_train = (
        ds_train.shuffle(1024)
        .map(map_func, num_parallel_calls=AUTO)
        .batch(config_model.getint('BATCH_SIZE'))
        .prefetch(AUTO) )
    
    #----------------------------------------------------------------------------------
    model_dir =  config_model.get('MODEL_DIR')    
    if not config_model.get('EXP_CODE') is None :
        model_dir = os.path.join(model_dir, config_model.get('EXP_CODE'))
    model_dir = os.path.join(model_dir, dataset_name, model_name)
    if not os.path.exists(model_dir) :
        os.makedirs(os.path.join(model_dir, 'ckp'))
        os.makedirs(os.path.join(model_dir, 'model'))
        print('--- {} was created'.format(os.path.dirname(model_dir)))
    #----------------------------------------------------------------------------------
    
    if gpu_id >= 0 :
        with tf.device('/device:GPU:{}'.format(gpu_id)) :
            model = siamese.Siamese(config_model, config_data)
            model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9))                            
            history = model.fit(ds_train, epochs = config_model.getint('EPOCHS'))
        
        # saving the final model    
        model_file = os.path.join(model_dir, 'model')
        history_file = os.path.join(model_dir, 'model', 'history.npy')
        np.save(history_file, history.history)
        model.save_weights(model_file)
        model.encoder.save(model_file)
        print('model was saved at {}'.format(model_file))        