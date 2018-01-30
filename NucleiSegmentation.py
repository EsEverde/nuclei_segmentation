# Imports and funcs
import tensorflow as tf
import os
from glob import glob


IMAGE_PATH = "/Users/ever/Desktop/Test/images"
MASK_PATH = "/Users/ever/Desktop/Test/masks"

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100

BATCH_SIZE = 20


#batch functions for Dataset
def TF_MinMaxScaler(Stack):
    Stack = tf.cast(Stack,tf.float32)

    Min = tf.reduce_min(Stack)
    Stack = tf.subtract(Stack, Min)

    Max = tf.reduce_max(Stack)
    MinMaxStack = tf.div(Stack, Max)

    return MinMaxStack

def image_import(value):
    image_file = tf.read_file(value)
    image = tf.image.decode_png(image_file, channels=3)
    image = tf.image.resize_images(image, size = [IMAGE_WIDTH,IMAGE_HEIGHT])
    image = TF_MinMaxScaler(image)

    #stack = TF_MaxAbsScaler(stack)
    return image

def mask_import(value):
    mask_file = tf.read_file(value)
    mask = tf.image.decode_png(mask_file, channels=3)
    mask = TF_MinMaxScaler(mask)
    #stack = TF_MaxAbsScaler(stack)
    return mask

def image_crop_to_size(image_source, image_destination):
    # crops 3 spatial dimensions in stack_source down to the size of stack_destination, symmetrically eroding exceeding pixels from the edges
    # both stacks have the shape [batches,width, height, depth, channels]

    source_shape = tf.shape(image_source)
    destination_shape = tf.shape(image_destination)

    width_diff = destination_shape[1]
    height_diff = destination_shape[2]

    start_width = source_shape[1] // 2 - (width_diff // 2)
    start_height = source_shape[2] // 2 - (height_diff // 2)

    image_crop = image_source[:, start_width:start_width + width_diff,
                 start_height:start_height + height_diff,:]

    # image_crop = tf.slice(image_source, begin = [0,width_diff,height_diff,depth_diff,0], size = new_shape)
    return image_crop


#build datasets

glob_pattern = os.path.join(IMAGE_PATH, '*')
image_files = glob(glob_pattern)

glob_pattern = os.path.join(MASK_PATH, '*')
mask_files = glob(glob_pattern)

images_train = tf.data.Dataset.from_tensor_slices(image_files)
images_train = images_train.map(image_import)
images_train = images_train.batch(BATCH_SIZE)
image_iterator = tf.data.Iterator.from_structure(images_train.output_types,images_train.output_shapes)
image_iterator_init = image_iterator.make_initializer(images_train)

masks_train = tf.data.Dataset.from_tensor_slices(mask_files)
masks_train = masks_train.map(mask_import)
masks_train = masks_train.batch(BATCH_SIZE)
mask_iterator = tf.data.Iterator.from_structure(masks_train.output_types,masks_train.output_shapes)
mask_iterator_init = mask_iterator.make_initializer(masks_train)

x = image_iterator.get_next()
y_ = mask_iterator.get_next()

#build 2D convnet

conv1 = tf.layers.conv2d(inputs = x, filters = 12, kernel_size = [3,3], activation = tf.nn.relu )
pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2,2], strides = [2,2])
norm1 = tf.layers.batch_normalization(pool1, axis = 3)

conv2 = tf.layers.conv2d(inputs = pool1, filters = 24, kernel_size = [3,3], activation = tf.nn.relu )
pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2,2], strides = [2,2])
norm2 = tf.layers.batch_normalization(pool2, axis = 3)

conv3 = tf.layers.conv2d(inputs = pool2, filters = 48, kernel_size = [3,3], activation = tf.nn.relu )
pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size = [2,2], strides = [2,2])
norm3 = tf.layers.batch_normalization(pool3, axis = 3)

# ... now deconv
deconv1 = tf.layers.conv2d_transpose(inputs = norm3, filters = 96, kernel_size = [3,3], strides=[2,2])
sandwich1 = tf.concat([deconv1, image_crop_to_size(pool2, deconv1)], axis = 3)
dnorm1 = tf.layers.batch_normalization(sandwich1, axis = 3)

deconv2 = tf.layers.conv2d_transpose(inputs = sandwich1, filters = 48, kernel_size = [3,3], strides=[2,2])
sandwich2 = tf.concat([deconv2, image_crop_to_size(pool1, deconv2)], axis = 3)
dnorm2 = tf.layers.batch_normalization(sandwich2, axis = 3)

deconv3 = tf.layers.conv2d_transpose(inputs = sandwich2, filters = 24, kernel_size = [3,3], strides = [2,2])
sandwich3 = tf.concat([deconv3, image_crop_to_size(x, deconv3)], axis = 3)
dnorm3 = tf.layers.batch_normalization(sandwich3, axis = 3)

# mini nn reduction
mininet1 = tf.layers.conv2d(inputs = dnorm3, filters = 12, kernel_size = [1,1], activation=tf.nn.relu )
mininet2 = tf.layers.conv2d(inputs = mininet1, filters = 8, kernel_size = [1,1], activation=tf.nn.relu )
mininet3 = tf.layers.conv2d(inputs = mininet2, filters = 4, kernel_size = [1,1], activation=tf.nn.relu )
regresion = tf.layers.conv2d(inputs = mininet3, filters = 3, kernel_size = [1,1], activation = tf.nn.sigmoid)
y_crop = image_crop_to_size(y_, regresion)

# loss func
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = y_crop, logits = regresion)
loss = tf.reduce_mean(xentropy)

# gradient descent optimizer
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)

# evaluation
correct = tf.equal(tf.round(regresion), y_crop)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# exec phase
init = tf.global_variables_initializer()
saver = tf.train.Saver()
n_epochs = 2

with tf.Session() as sess:
    init.run()

    for iteration in range(n_epochs):
        sess.run(image_iterator_init)
        sess.run(mask_iterator_init)
        acc_train = 0
        while True:
            try:
                sess.run(training_op)
            except tf.errors.OutOfRangeError:
                break

        print('Epoch: ' + str(iteration))
        print('  -=-------=-  ')

    save_path=saver.save(sess, "./my_model.ckpt")
