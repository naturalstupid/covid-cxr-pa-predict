import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet201, ResNet101V2
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
#### Classes
class DataProcessor():
    def __init__(self, data_location):
        self.labeled_dataset = tf.data.Dataset.list_files(f"{data_location}/*/*")
        
    def _get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-2] == CLASS_NAMES
    
    def _decode_image(self, img):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [IMAGE_SHAPE[0], IMAGE_SHAPE[1]])
    
    def _pre_process_images(self, file_path):
        label = self._get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self._decode_image(img)
        return [img, label]
    
    def prepare_dataset(self):
        self.labeled_dataset = self.labeled_dataset.map(self._pre_process_images)
        self.labeled_dataset = self.labeled_dataset.cache()
        self.labeled_dataset = self.labeled_dataset.shuffle(buffer_size=10)
        self.labeled_dataset = self.labeled_dataset.repeat()
        self.labeled_dataset = self.labeled_dataset.batch(BATCH_SIZE)
        self.labeled_dataset = self.labeled_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        #print(self.labeled_dataset)
        train_size = int(TRAIN_FRACTION * DATASET_SIZE)
        print('Training size',train_size)
        val_size = int(VALIDATION_FRACTION * DATASET_SIZE)
        print('Validation size',val_size)
        test_size = int(TEST_FRACTION * DATASET_SIZE)
        print('Test size',test_size)
        
        train_dataset = self.labeled_dataset.take(train_size)
        test_dataset = self.labeled_dataset.skip(train_size)
        val_dataset = test_dataset.skip(test_size)
        test_dataset = test_dataset.take(test_size)
        
        return train_dataset, test_dataset, val_dataset

class Wrapper(tf.keras.Model):
    def __init__(self, base_model):
        super(Wrapper, self).__init__()
        
        self.base_model = base_model
        self.average_pooling_layer = AveragePooling2D(name="polling")
        self.flatten = Flatten(name="flatten")
        self.dense = Dense(64, activation="relu")
        self.dropout = Dropout(0.5)
        self.output_layer = Dense(len(CLASS_NAMES), activation="softmax")
        
    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.average_pooling_layer(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x)
        output = self.output_layer(x)
        return output
start_time = time.time()
data_location = './data'
CLASS_NAMES = np.array([dir for r, dir, f in os.walk(data_location+'/') if dir != ''])
CLASS_NAMES = [x for x in CLASS_NAMES if x != []]
CLASS_NAMES = [item for sublist in CLASS_NAMES for item in sublist]
print(CLASS_NAMES, len(CLASS_NAMES))
#exit()
#CLASS_NAMES = ['covid-19', 'healthy']
IMAGE_SHAPE = (256, 256, 3)
BATCH_SIZE = 8
EPOCHS = 10
DATASET_SIZE = sum([len(files) for r, d, files in os.walk(data_location)])
print ('DATASET_SIZE',DATASET_SIZE)
use_dense_net = True
use_mobile_net = False
use_res_net = False
run_saved_model = True
TRAIN_FRACTION = 0.70
TEST_FRACTION = 0.15
VALIDATION_FRACTION = 0.15
METRICS = ['accuracy']# ['binary_accuracy', 'categorical_accuracy']
LOSS_METHOD = 'binary_crossentropy'#'categorical_crossentropy' #'binary_crossentropy'

processor = DataProcessor(data_location)
train_dataset, test_dataset, val_dataset = processor.prepare_dataset()
#print (test_dataset)
#exit()
base_learning_rate = 0.0001
steps_per_epoch = DATASET_SIZE//BATCH_SIZE
validation_steps = 20
if use_mobile_net and not(run_saved_model):    
    mobile_net = MobileNetV2(input_shape=IMAGE_SHAPE, include_top=False, weights='imagenet')
    mobile_net.trainable = False
    mobile = Wrapper(mobile_net)
    mobile.compile(Adam(lr=base_learning_rate),
                  loss=LOSS_METHOD,
                  metrics=METRICS)
if use_res_net and not(run_saved_model):
    res_net = ResNet101V2(input_shape=IMAGE_SHAPE, include_top=False, weights='imagenet')
    res_net.trainable = False
    res = Wrapper(res_net)
    res.compile(optimizer=Adam(lr=base_learning_rate),
                  loss=LOSS_METHOD,
                  metrics=METRICS)
if use_dense_net and not(run_saved_model):
    dense_net = DenseNet201(input_shape=IMAGE_SHAPE, include_top=False, weights='imagenet')
    dense_net.trainable = False
    dense = Wrapper(dense_net)
    dense.compile(optimizer=Adam(lr=base_learning_rate),
                  loss=LOSS_METHOD,
                  metrics=METRICS)
if use_mobile_net and not(run_saved_model):
    history_mobile = mobile.fit(train_dataset,
                        epochs=EPOCHS,
                        validation_data=val_dataset,
                        validation_steps=validation_steps)
if use_res_net and not(run_saved_model):
    history_resnet = res.fit(train_dataset,
                        epochs=EPOCHS,
                        validation_data=val_dataset,
                        validation_steps=validation_steps)
if use_dense_net and not(run_saved_model):
    history_densenet = dense.fit(train_dataset,
                        epochs=EPOCHS,
                        validation_data=val_dataset,
                        validation_steps=validation_steps)
if use_mobile_net and not(run_saved_model):
    plt.plot(history_mobile.history['accuracy'])
    plt.plot(history_mobile.history['val_accuracy'])
    plt.title('Model accuracy - Mobile Net')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history_mobile.history['loss'])
    plt.plot(history_mobile.history['val_loss'])
    plt.title('Model loss - Mobile Net')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
if use_res_net and not(run_saved_model):
    plt.plot(history_resmet.history['accuracy'])
    plt.plot(history_resnet.history['val_accuracy'])
    plt.title('Model accuracy - ResNet')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    plt.plot(history_resnet.history['loss'])
    plt.plot(history_resnet.history['val_loss'])
    plt.title('Model loss - ResNet')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
if use_dense_net and not(run_saved_model):
    plt.plot(history_densenet.history['accuracy'])
    plt.plot(history_densenet.history['val_accuracy'])
    plt.title('Model accuracy - Dense Net')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    plt.plot(history_densenet.history['loss'])
    plt.plot(history_densenet.history['val_loss'])
    plt.title('Model loss - DenseNet')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
if  run_saved_model:
    print ('Loading saved model')
    if use_mobile_net:
        print('Retrieving model:mobile from ./models/mobilenet/')
        #mobile = tf.saved_model.load('./models/mobilenet/1')
        mobile = tf.keras.models.load_model('./models/mobilenet/')
    if use_res_net:
        print('Retrieving model:resnet from ./models/resnet/')
        #res = tf.saved_model.load('./models/resnet/1')
        res = tf.keras.models.load_model('./models/resnet/')
    if use_dense_net:
        print('Retrieving model:dense from ./models/densenet/')
        #dense = tf.saved_model.load('./models/densenet/1')
        dense = tf.keras.models.load_model('./models/densenet/')
if use_mobile_net:
    loss, accuracy = mobile.evaluate(test_dataset, steps = validation_steps)
    Y_pred = mobile.predict(test_dataset)#, test_size)#, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)
    y_pred = np.argmax(Y_pred,axis=1)
    
    print("--------MobileNet---------")
    print("Loss: {:.2f}".format(loss))
    print("Accuracy: {:.2f}".format(accuracy))
    print("---------------------------")
    if not(run_saved_model):
        print('Saving model:mobilenet to ./models/mobilenet/')
        #tf.saved_model.save(mobile, './models/mobilenet/1')
        mobile.save('./models/mobilenet/')
if use_res_net:
    loss, accuracy = res.evaluate(test_dataset, steps = validation_steps)
    Y_pred = res.predict(test_dataset)#, test_size)#, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)
    y_pred = np.argmax(Y_pred,axis=1)
    
    print("--------ResNet---------")
    print("Loss: {:.2f}".format(loss))
    print("Accuracy: {:.2f}".format(accuracy))
    print("---------------------------")
    if not(run_saved_model):
        print('Saving model:resnet to ./models/resnet/')
        #tf.saved_model.save(res, './models/resnet/1')
        res.save('./models/resnet/')
if use_dense_net:
    loss, accuracy = dense.evaluate(test_dataset, steps = validation_steps)
    end_time = time.time()
    print('elapsed time',end_time-start_time)
    Y_pred = dense.predict(test_dataset)#, test_size)#, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)
    y_pred = np.argmax(Y_pred,axis=1)
#    print(y_pred)
#    print(confusion_matrix(x_pred, y_pred))#, CLASS_NAMES))#, sample_weight))
    print("--------DenseNet---------")
    print("Loss: {:.2f}".format(loss))
    print("Accuracy: {:.2f}".format(accuracy))
    print("---------------------------")
    if not(run_saved_model):
        print('Saving model:dense to ./models/densenet/')
        #tf.saved_model.save(dense, './models/densenet/1')
        dense.save('./models/densenet/')
end_time = time.time()
print('elapsed time',end_time-start_time)
