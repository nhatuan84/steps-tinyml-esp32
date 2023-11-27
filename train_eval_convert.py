import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
import albumentations as A
import glob
import numpy as np
import random
from sklearn.metrics import confusion_matrix, accuracy_score

EPOCHS = 100
NUM_CLASSES = 2
BATCH_SIZE = 35
STEPS_PER_EPOCH = 1001//BATCH_SIZE
VAL_STEPS = 501//BATCH_SIZE
IMAGE_SIZE = 96
PATH = r'/content/drive/MyDrive/Colab Notebooks/'

classes = {'cat':0, 'dog':1}
samples = []

transforms = A.Compose([
    A.OneOf([
            A.GaussNoise(),
       ], p=0.2),
    A.RandomBrightnessContrast(p=0.2),
    A.Transpose(),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
    A.Blur(blur_limit=3),
    A.OpticalDistortion(),
    A.GridDistortion(),
    A.HueSaturationValue(),
    A.CLAHE(),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(),
    A.augmentations.geometric.rotate.Rotate()
])

def one_hot(label):
    a = np.zeros(len(classes))
    a[label] = 1
    return a

def get_data(t):
  data_folders = ['data_cd/{}/cats'.format(t),
                  'data_cd/{}/dogs'.format(t)]
  return data_folders

def image_resize(img, size=(IMAGE_SIZE,IMAGE_SIZE), keep_aspect=False):
    if keep_aspect == False:
        return cv2.resize(img, size, cv2.INTER_AREA)
    else:
        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape)>2 else 1
        if h == w:
            return cv2.resize(img, size, cv2.INTER_AREA)
        dif = h if h > w else w
        interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC
        x_pos = (dif - w)//2
        y_pos = (dif - h)//2
        if len(img.shape) == 2:
            mask = np.zeros((dif, dif), dtype=img.dtype) + np.random.randint(255)
            mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
        else:
            mask = np.zeros((dif, dif, c), dtype=img.dtype) + np.random.randint(255)
            mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
        return cv2.resize(mask, size, interpolation)

def load_data(t, samples):
    for folder in get_data(t):
      images = glob.glob(PATH + folder + '/*')
      for image in images:
        samples.append(image)
    random.shuffle(samples)

def gen_batch(t):
    samples = []
    loop = 1
    while loop:
        objs = []
        labels = []
        if len(samples) == 0:
          load_data(t, samples)
        for i in range(BATCH_SIZE):
          if len(samples) > 0:
            path = samples.pop(0)
            image = cv2.imread(path)
            image = image_resize(image)
            image = transforms(image=image)['image']/255.0
            objs.append(image)
            for c in classes:
              if c in path:
                labels.append(one_hot(classes[c]))
        yield (np.asarray(objs), np.asarray(labels))

train_data = gen_batch('train')
val_data = gen_batch('validation')

base_model = tf.keras.applications.MobileNet(   alpha=0.25,
                                                input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),
                                                weights="imagenet",
                                                include_top=False)

print("Number of layers in the base model: ", len(base_model.layers))
base_model.trainable = True
fine_tune_at = 0*len(base_model.layers)//3
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, name='output')

inputs = tf.keras.Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3), name='input')
outputs = base_model(inputs, training=False)
outputs = global_average_layer(outputs)
outputs = tf.keras.layers.Dropout(0.2)(outputs)
outputs = prediction_layer(outputs)
model = tf.keras.Model(inputs, outputs)

#tf.keras.utils.plot_model(model, show_shapes=True)
model.summary()

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=PATH+'trained_cd/model.h5',
    #save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(train_data,
                  steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
                  validation_steps=VAL_STEPS, validation_data=val_data,
                  callbacks=[model_checkpoint_callback])


def convert_stats():
    samples = []
    load_data('validation', samples)
    i=0
    while True:
      if len(samples) > 0:
        path = samples.pop(0)
        image = cv2.imread(path)
        image = image_resize(image)
        image = transforms(image=image)['image']/255.0
        i += 1
        if i == 100:
          break
        yield {'input':np.expand_dims(image, axis=0).astype(np.float32)}

model = tf.keras.models.load_model(PATH+'trained_cd/model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# which include quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = convert_stats
# Restricting supported target op specification to INT8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
# Save the model to disk
open(PATH+'trained_cd/quantized.tflite', "wb").write(tflite_model)

model_quantized = tf.lite.Interpreter(PATH+'trained_cd/quantized.tflite')
model_quantized.allocate_tensors()

input_scales = model_quantized.get_input_details()[0]['quantization_parameters']['scales']
input_zero_points = model_quantized.get_input_details()[0]['quantization_parameters']['zero_points']

output_scales = model_quantized.get_output_details()[0]['quantization_parameters']['scales']
output_zero_points = model_quantized.get_output_details()[0]['quantization_parameters']['zero_points']

model_quantized_input_index = model_quantized.get_input_details()[0]["index"]
model_quantized_output_index = model_quantized.get_output_details()[0]["index"]


model_quantized_predictions = []
labels = []
samples = []
load_data('validation', samples)
label = 0
for path in samples:
  image = cv2.imread(path)
  image = image_resize(image)
  #image = transforms(image=image)['image']/255.0
  #image = image/input_scales + input_zero_points
  for c in classes:
    if c in path:
      label = classes[c]
  x_value_tensor = tf.convert_to_tensor([image.astype(np.uint8)], dtype=tf.uint8)
  model_quantized.set_tensor(model_quantized_input_index, x_value_tensor)
  model_quantized.invoke()
  output = (model_quantized.get_tensor(model_quantized_output_index)[0] - output_zero_points) * output_scales
  model_quantized_predictions.append(np.argmax(output))
  labels.append(label)

print(confusion_matrix(labels, model_quantized_predictions))
print(accuracy_score(labels, model_quantized_predictions))

#convert model to code
#sudo apt-get  install xxd
#xxd -i quantized.tflite > quantized.cc
