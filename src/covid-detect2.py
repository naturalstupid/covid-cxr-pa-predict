import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, MobileNetV2, DenseNet201, ResNet101V2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import time
from sklearn.preprocessing.label import LabelEncoder
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          save_as_file=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if save_as_file is None:
        plt.show()
    else:
        plt.savefig(save_as_file)
def get_command_line_arguments():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default="../data", #required=True,
        help="path to input dataset")
    ap.add_argument("-g", "--graph", type=str, default="plot.png",
        help="path to output loss/accuracy plot")
    ap.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL_NAME,
        help="keras model name to be used to train the data: Choose one of "+','.join(AVAILABLE_MODELS.keys()))
    ap.add_argument("-o", "--output", type=str, default="covid19.model",
        help="path to save model data - default=covid19-model")
    ap.add_argument("-p", "--predict", type=str, default="",#../pneumonia-sample.jpeg",
        help="path to save model data - default=covid19-model")
    return vars(ap.parse_args())
def process_image(imagePath):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    image = image / 255.0
    return image
def get_data_and_labels_from_image_path(imagePaths):
    data = []
    labels = []
    file_names = []
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        file_name = label + '-' + os.path.basename(imagePath)
        image = process_image(imagePath)
        data.append(image)
        labels.append(label)
        file_names.append(file_name)
    data = np.array(data)
    labels = np.array(labels)
    file_names = np.array(file_names)
    lb = LabelEncoder()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    return [data,labels, lb.classes_, file_names]
def get_model(model_name):
    if model_name in AVAILABLE_MODELS:
        baseModel = AVAILABLE_MODELS[model_name](input_shape=IMAGE_SHAPE, include_top=False, weights='imagenet')
    else:
        baseModel = AVAILABLE_MODELS[DEFAULT_MODEL_NAME](input_shape=IMAGE_SHAPE, include_top=False, weights='imagenet')
    headModel = baseModel.output
    headModel = AveragePooling2D(name="polling")(headModel)#(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(CLASS_SIZE, activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    for layer in baseModel.layers:
        layer.trainable = False
    return model
def show_training_validation_accuracy_loss(H,plot_file_name):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on COVID-19 Dataset("+model_name+')')
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plot_file_name)
def show_predict_image_from_test_set(number_of_random_samples, show_image=False):
    import random
    ids = random.sample(range(1, test_size+1), number_of_random_samples)
    print('number_of_random_samples',number_of_random_samples,'ids size',len(ids))
    for id in ids:
        test_img = testX[id-1]
        reshaped_img = test_img.reshape((1,IMAGE_SHAPE[0],IMAGE_SHAPE[1],IMAGE_SHAPE[2]))
        prediction = model.predict(reshaped_img)
        predicted_label = CLASS_CATEGORIES[np.argmax(prediction)] 
        print('image-id',id-1,'Prediction Score',prediction,'predicted label', predicted_label)
        if (show_image):
            plt.imshow(test_img)
            plt.title(predicted_label + ' id='+str(id))
            plt.show()
def show_predict_image(imagePath, show_image=False):  
    test_img = process_image(os.path.abspath(imagePath))
#    print('test image shape',test_img.shape)
    reshaped_img = test_img.reshape((1,IMAGE_SHAPE[0],IMAGE_SHAPE[1],IMAGE_SHAPE[2]))
    prediction = model.predict(reshaped_img)
    predicted_label = CLASS_CATEGORIES[np.argmax(prediction)] 
    print(imagePath,'Prediction Score',prediction,'predicted label', predicted_label)
    if (show_image):
        plt.imshow(test_img)
        plt.title(predicted_label)
        plt.show()
#def get_area_under_roc_curve():
    
####   Main script starts here 
start_time = time.time() 
load_trained_model = True
AVAILABLE_MODELS={'VGG':VGG16, 'MobileNet':MobileNetV2, 'DenseNet':DenseNet201, 'ResNet':ResNet101V2}
DEFAULT_MODEL_NAME = 'MobileNet'
args = get_command_line_arguments()
data_path = args["dataset"]
model_name = args["model"]
plot_file_name = '../results/'+model_name+'-'+args["graph"]
save_file_name = '../models/'+model_name+'-'+args["output"]
predict_image_path = os.path.abspath(args['predict'])
print(predict_image_path)
IMAGE_SHAPE = (256, 256, 3)
cm_file_name = '../results/'+model_name+'-'+'confusion_matrix.png'
cm_title = model_name+'-'+'Confusion matrix'
INIT_LR = 1e-3
EPOCHS = 25
BS = 8
IMAGE_SHAPE = (256, 256, 3)
TEST_SIZE_FRACTION = 0.20
print("[INFO] loading images...")
imagePaths = list(paths.list_images(data_path))
[data,labels, CLASS_CATEGORIES, file_names] = get_data_and_labels_from_image_path(imagePaths)
data_size = len(data)
test_size = int(TEST_SIZE_FRACTION*data_size)
train_size = int((1.0-TEST_SIZE_FRACTION)*data_size)
CLASS_SIZE = len(CLASS_CATEGORIES)
print('data size',data_size,'train size',train_size,'#categories',CLASS_CATEGORIES)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=TEST_SIZE_FRACTION, stratify=labels, random_state=42)
print('textX shape',testX.shape)
print("Model:",model_name)
if (load_trained_model):
    if (os.path.exists(save_file_name)):
        print(" Retrieving model:",save_file_name)
        model = tf.keras.models.load_model(save_file_name)
    else:
        print("Trained Model File:"+save_file_name+' not found')
        exit()
else:    
    model = get_model(model_name)
    # compile our model
    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    print("[INFO] training head...")
    H = model.fit(trainX, trainY, epochs=EPOCHS, validation_data=(testX, testY), validation_steps=len(testX) // BS)
predict_image_path = ""
if (predict_image_path == ""):    
    print("[INFO] evaluating network...")
    Y_Predicted = model.predict(testX, batch_size=BS)
    probs = Y_Predicted[:,1]
    Y_Predicted = np.argmax(Y_Predicted, axis=1)
    print(metrics.classification_report(testY.argmax(axis=1), Y_Predicted, target_names=CLASS_CATEGORIES))
    cm = metrics.confusion_matrix(testY.argmax(axis=1), Y_Predicted)
    plot_confusion_matrix(cm,CLASS_CATEGORIES,save_as_file=cm_file_name,title=cm_title)
    # Compute the AUC Score.
    auc = metrics.roc_auc_score(testY, probs)
    print('AUC: %.2f' % auc)
    # Get the ROC Curve.
    fpr, tpr, thresholds = metrics.roc_curve(testY, probs)
    metrics.plot_roc_curve(fpr, tpr)
    
if (not(load_trained_model)):
    show_training_validation_accuracy_loss(H,plot_file_name)
end_time = time.time()
print('cpu time for model.train/evaluate',end_time-start_time)
start_time=end_time
if (not(load_trained_model)):
    print("[INFO] saving COVID-19 detector model...")
    model.save(save_file_name, save_format="h5")
    end_time = time.time()
    print('cpu time for saving model',end_time-start_time)
# Predict an image
show_predict_image("../covid-kjr-21-e24-g001-l-a.jpg")
show_predict_image("../pneumonia-person25_bacteria_118.jpeg")
show_predict_image("../healthy-NORMAL2-IM-1345-0001.jpeg")
