# %%
#1. Import packages
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow import keras

from IPython.display import clear_output

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os, cv2

# %%
#2. Data preparation
#2.1 Prepare the path
root_path = os.path.join(os.getcwd(), 'data-science-bowl-2018-2', 'train')
#root_path = r"C:\Users\user\Desktop\ml\Image_Segmentation\data-science-bowl-2018-2\train"
#root_path2 = os.path.join(os.getcwd(), 'test')

# %%
#2.2 Prepare empty list to hold the data
images = []
masks = []

# %%
#2.3 Load the images using opencv
image_dir = os.path.join(root_path, 'inputs')
for image_file in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir, image_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128,128))
    images.append(img)

# %%
#2.4 Load the masks
masks_dir = os.path.join(root_path, 'masks')
for mask_file in os.listdir(masks_dir):
    mask = cv2.imread(os.path.join(masks_dir, mask_file), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128, 128))
    masks.append(mask)
    
# %%
#2.5 Convert the list of np array into a np array
images_np = np.array(images)
masks_np = np.array(masks)

# %%
#3. Data preprocessing
#3.1 Expand the mask dimension
masks_np_exp = np.expand_dims(masks_np, axis =-1)

#Check the mask output
print(np.unique(masks_np_exp[0]))

# %%
#3.2 Convert the mask values from [0,255] into [0,1]
converted_masks = np.round(masks_np_exp / 255.0).astype(np.int64)

#Check the mask output
print(np.unique(converted_masks[0]))

# %%
#3.3 Normalize the images
converted_images = images_np / 255.0

# %%
#4. Perform train test split
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(converted_images, converted_masks, test_size=0.2, random_state=SEED)

# %%
#5. Convert the numpy arrays into tensor slices
X_train_tensor = tf.data.Dataset.from_tensor_slices(X_train)
X_test_tensor = tf.data.Dataset.from_tensor_slices(X_test)
y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
y_test_tensor = tf.data.Dataset.from_tensor_slices(y_test)

# %%
#6. Combine the images and masks using zip method
train_dataset = tf.data.Dataset.zip((X_train_tensor, y_train_tensor))
test_dataset = tf.data.Dataset.zip((X_test_tensor, y_test_tensor))

# %%
#7. Define data augmentation pipeline through subclassing
class Augment(keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_inputs = keras.layers.RandomFlip(mode='horizontal', seed=seed)
        self.augment_labels = keras.layers.RandomFlip(mode='horizontal', seed=seed)

    def __call__(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels
    
# %%
#8. Build the dataset  
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
TRAIN_SIZE = len(train_dataset)
STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE
train_batches = (
    train_dataset
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_batches = test_dataset.batch(BATCH_SIZE)

# %%
#9. Visualize some pictures as example
def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
    plt.show()

for images, masks in train_batches.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])
    
# %%
#10. Model Development
#10.1 Use a pretrained model as the feature extractor
base_model = keras.applications.MobileNetV2(input_shape=[128,128,3], include_top=False)
base_model.summary()

# %%
#10.2 Use these activation layers as the outputs from the feature extractor (some of these output will be used to perform concatenation at the upsampling path)
layer_names =  [
    'block_1_expand_relu',  #64x64
    'block_3_expand_relu',  #32x32
    'block_6_expand_relu',  #16x16
    'block_13_expand_relu', #8x8
    'block_16_project'      #4x4
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# %%
#10.3 Instantiate the feature extractor
down_stack = keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

# %%
#10.4 Define the upsampling path 
up_stack = [
    pix2pix.upsample(512, 3),  #(num of nodes, filter size)  4x4 --> 8x8
    pix2pix.upsample(256, 3), #8x8 --> 16x16
    pix2pix.upsample(128, 3),  #16x16 --> 32x32
    pix2pix.upsample(64, 3) #32x32 --> 64x64
]

# %%
#10.5 Use functional API to construct the entire U-net

def unet(output_channels:int):
    inputs = keras.layers.Input(shape=(128,128,3))
    #Downsample through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    #Build the upsampling path and establish the concatenation
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = keras.layers.Concatenate()
        x = concat([x, skip])

    #Use a transpose convolution layer to perform the last unsampling, this will become the output layer
    last = keras.layers.Conv2DTranspose(filters=output_channels, kernel_size=3, strides=2, padding='same') #64x64 --> 128x128
    outputs = last(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# %%
#10.6 Use the function to create the model
OUTPUT_CHANNELS = 3
model = unet(OUTPUT_CHANNELS)
model.summary()
keras.utils.plot_model(model)

# %%
#11. Compile the model
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

# %%
#12. Create functions to show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])

show_predictions()

# %%
#13. Create a callback function to make use of the show_predictions function
class Displaycallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample prediction after epoch {}\n'.format(epoch+1))

# %%
#14. Create tensorboard callback
log_path = os.path.join('log_dir', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = TensorBoard(log_dir = log_path)
es = EarlyStopping(monitor = 'accuracy', patience = 5, verbose = 0, restore_best_weights = True)

# %%
#15. Model training
EPOCHS = 10
VAL_SUBSPLITS = 10
VALIDATION_STEPS = len(test_dataset) // BATCH_SIZE // VAL_SUBSPLITS
history  = model.fit(train_batches, validation_data=test_batches , validation_steps=VALIDATION_STEPS, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, callbacks=[Displaycallback(),tb, es])

# %%
#16. Model deployment 
show_predictions(test_batches, 3)

# %%
#17. Save model
model.save('model.h5')



# %%

#  TESTING(FOLDER TEST)
#2. Data preparation
#2.1 Prepare the path
root_path2 = os.path.join(os.getcwd(), 'data-science-bowl-2018-2', 'test')

# %%
#2.2 Prepare empty list to hold the data
images2 = []
masks2 = []

# %%
#2.3 Load the images using opencv
image_dir2 = os.path.join(root_path2, 'inputs')
for image_file2 in os.listdir(image_dir2):
    img2 = cv2.imread(os.path.join(image_dir2, image_file2))
    img2= cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, (128,128))
    images2.append(img2)

# %%
#2.4 Load the masks
masks_dir2 = os.path.join(root_path2, 'masks')
for mask_file2 in os.listdir(masks_dir2):
    mask2 = cv2.imread(os.path.join(masks_dir2, mask_file2), cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.resize(mask2,(128, 128))
    masks2.append(mask2)
    
# %%
#2.5 Convert the list of np array into a np array
images_np2 = np.array(images2)
masks_np2 = np.array(masks2)

# %%
#3. Data preprocessing
#3.1 Expand the mask dimension
masks_np_exp2 = np.expand_dims(masks_np2, axis =-1)

#Check the mask output
print(np.unique(masks_np_exp2[0]))

# %%
#3.2 Convert the mask values from [0,255] into [0,1]
converted_masks2 = np.round(masks_np_exp2 / 255.0).astype(np.int64)

#Check the mask output
print(np.unique(converted_masks2[0]))

# %%
#3.3 Normalize the images
converted_images2 = images_np2 / 255.0

# %%
#4. Perform train test split
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(converted_images2, converted_masks2, test_size=0.2, random_state=SEED)

# %%
#5. Convert the numpy arrays into tensor slices
X_train_tensor2 = tf.data.Dataset.from_tensor_slices(X_train)
X_test_tensor2 = tf.data.Dataset.from_tensor_slices(X_test)
y_train_tensor2 = tf.data.Dataset.from_tensor_slices(y_train)
y_test_tensor2 = tf.data.Dataset.from_tensor_slices(y_test)

# %%
#6. Combine the images and masks using zip method
train_dataset2 = tf.data.Dataset.zip((X_train_tensor2, y_train_tensor2))
test_dataset2 = tf.data.Dataset.zip((X_test_tensor2, y_test_tensor2))

# %%
#7. Define data augmentation pipeline through subclassing
class Augment(keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_inputs = keras.layers.RandomFlip(mode='horizontal', seed=seed)
        self.augment_labels = keras.layers.RandomFlip(mode='horizontal', seed=seed)

    def __call__(self, inputs2, labels2):
        inputs2 = self.augment_inputs(inputs2)
        labels2 = self.augment_labels(labels2)
        return inputs2, labels2
    
# %%
#8. Build the dataset  
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
TRAIN_SIZE = len(train_dataset2)
STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE
train_batches = (
    train_dataset2
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_batches = test_dataset2.batch(BATCH_SIZE)

# %%
#9. Visualize some pictures as example
def display(display_list2):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list2)):
        plt.subplot(1,len(display_list2), i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list2[i]))
    plt.show()

for images, masks in train_batches.take(5):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])
    
# %%
#10. Model Development
#10.1 Use a pretrained model as the feature extractor
base_model2 = keras.applications.MobileNetV2(input_shape=[128,128,3], include_top=False)
base_model2.summary()

# %%
#10.2 Use these activation layers as the outputs from the feature extractor (some of these output will be used to perform concatenation at the upsampling path)
layer_names2 =  [
    'block_1_expand_relu',  #64x64
    'block_3_expand_relu',  #32x32
    'block_6_expand_relu',  #16x16
    'block_13_expand_relu', #8x8
    'block_16_project'      #4x4
]

base_model_outputs2 = [base_model2.get_layer(name).output for name in layer_names2]

# %%
#10.3 Instantiate the feature extractor
down_stack2 = keras.Model(inputs=base_model2.input, outputs=base_model_outputs2)
down_stack2.trainable = False

# %%
#10.4 Define the upsampling path 
up_stack2 = [
    pix2pix.upsample(512, 3),  #(num of nodes, filter size)  4x4 --> 8x8
    pix2pix.upsample(256, 3), #8x8 --> 16x16
    pix2pix.upsample(128, 3),  #16x16 --> 32x32
    pix2pix.upsample(64, 3) #32x32 --> 64x64
]

# %%
#10.5 Use functional API to construct the entire U-net

def unet(output_channels:int):
    inputs2 = keras.layers.Input(shape=(128,128,3))
    #Downsample through the model
    skips2 = down_stack2(inputs2)
    x2 = skips2[-1]
    skips2 = reversed(skips2[:-1])

    #Build the upsampling path and establish the concatenation
    for up2, skip2 in zip(up_stack2, skips2):
        x2 = up2(x2)
        concat2 = keras.layers.Concatenate()
        x2 = concat2([x2, skip2])

    #Use a transpose convolution layer to perform the last unsampling, this will become the output layer
    last2 = keras.layers.Conv2DTranspose(filters=output_channels, kernel_size=3, strides=2, padding='same') #64x64 --> 128x128
    outputs2 = last2(x2)

    model2 = keras.Model(inputs=inputs2, outputs=outputs2)

    return model2

# %%
#10.6 Use the function to create the model
OUTPUT_CHANNELS = 3
model2 = unet(OUTPUT_CHANNELS)
model2.summary()
keras.utils.plot_model(model2)

# %%
#11. Compile the model
loss2 = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model2.compile(optimizer='adam', loss=loss2, metrics=['accuracy'])

# %%
#12. Create functions to show predictions
def create_mask(pred_mask2):
    pred_mask2 = tf.argmax(pred_mask2, axis=-1)
    pred_mask2 = pred_mask2[..., tf.newaxis]
    return pred_mask2[0]

def show_predictions(dataset=None, num=1):
    if dataset:
        for image2, mask2 in dataset.take(num):
            pred_mask2 = model2.predict(image2)
            display([image2[0], mask2[0], create_mask(pred_mask2)])
    else:
        display([sample_image, sample_mask, create_mask(model2.predict(sample_image[tf.newaxis, ...]))])

show_predictions()

# %%
#13. Create a callback function to make use of the show_predictions function
class Displaycallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample prediction after epoch {}\n'.format(epoch+1))

# %%
#14. Create tensorboard callback
log_path2 = os.path.join('log_dir_test', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb2 = callbacks.TensorBoard(log_dir = log_path2)

# %%
#15. Model training
EPOCHS = 10
VAL_SUBSPLITS = 10
VALIDATION_STEPS = len(test_dataset2) // BATCH_SIZE // VAL_SUBSPLITS
history2  = model2.fit(train_batches, validation_data=test_batches , validation_steps=VALIDATION_STEPS, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, callbacks=[Displaycallback(), tb2])

# %%
#16. Model deployment 
show_predictions(test_batches, 3)

