# import the necessary packages
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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
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
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
        help="path to output loss/accuracy plot")
    ap.add_argument("-m", "--model", type=str, default="MobileNetV2",
        help="keras model name to be used to train the data: Choose one of VGG16, MobileNetV2, DenseNet201, ResNet101V2")
    ap.add_argument("-o", "--output", type=str, default="covid19.model",
        help="path to save model data - default=covid19-model")
    return vars(ap.parse_args())
def get_data_and_labels_from_image_path(imagePaths):
    data = []
    labels = []
    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]
        # load the image, swap color channels, and resize it to be a fixed
        # 224x224 pixels while ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)
    # convert the data and labels to NumPy arrays while scaling the pixel
    # intensities to the range [0, 1]
    data = np.array(data) / 255.0
    labels = np.array(labels)
    return [data,labels]
def get_model(model_name):
    default_model = default_model_name
    if model_name in available_models:
        baseModel = available_models[model_name](input_shape=IMAGE_SHAPE, include_top=False, weights='imagenet')
    else:
        baseModel = available_models[default_model](input_shape=IMAGE_SHAPE, include_top=False, weights='imagenet')
    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(name="polling")(headModel)#(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(CLASS_SIZE, activation="softmax")(headModel)
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False
    return model

####   Main script starts here 
start_time = time.time() 
args = get_command_line_arguments()
# initialize the initial learning rate, number of epochs to train for,
# and batch size
data_path = args["dataset"]
model_name = args["model"]
plot_file_name = '../results/'+model_name+'-'+args["plot"]
save_file_name = '../models/'+model_name+'-'+args["output"]
cm_file_name = '../results/'+model_name+'-'+'confusion_matrix.png'
cm_title = model_name+'-'+'Confusion matrix'
available_models={'VGG16':VGG16, 'MobileNetV2':MobileNetV2, 'DenseNet201':DenseNet201, 'ResNet101V2':ResNet101V2}
default_model_name = 'DenseNet201'
#print('data path',data_path)
INIT_LR = 1e-3
EPOCHS = 25
BS = 8
IMAGE_SHAPE = (256, 256, 3)
TEST_SIZE_FRACTION = 0.20
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(data_path))
[data,labels] = get_data_and_labels_from_image_path(imagePaths)
data_size = len(data)
train_size = int((1.0-TEST_SIZE_FRACTION)*data_size)
# perform one-hot encoding on the labels
lb = LabelEncoder() #LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
CLASS_SIZE = len(lb.classes_)
print('data size',data_size,'train size',train_size,'#categories',lb.classes_)
#exit()
#print(labels)
#exit()
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=TEST_SIZE_FRACTION, stratify=labels, random_state=42)
print("Model:",model_name)
model = get_model(model_name)
# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
# train the head of the network
print("[INFO] training head...")
H = model.fit(trainX, trainY, epochs=EPOCHS, validation_data=(testX, testY), validation_steps=len(testX) // BS)
# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))
# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
#total = sum(sum(cm))
#acc = (cm[0, 0] + cm[1, 1]) / total
#sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
#specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# show the confusion matrix, accuracy, sensitivity, and specificity

plot_confusion_matrix(cm,lb.classes_,save_as_file=cm_file_name,title=cm_title)
#print("acc: {:.4f}".format(acc))
#print("sensitivity: {:.4f}".format(sensitivity))
#print("specificity: {:.4f}".format(specificity))
# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset("+model_name+')')
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(plot_file_name)
# serialize the model to disk
end_time = time.time()
print('cpu time for model.train/evaluate',end_time-start_time)
start_time=end_time
print("[INFO] saving COVID-19 detector model...")
model.save(save_file_name, save_format="h5")
end_time = time.time()
print('cpu time for saving model',end_time-start_time)
