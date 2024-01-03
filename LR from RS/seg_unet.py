from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from keras.layers import merge
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
# from tensorflow.keras import backend
from tensorflow import keras
# import losses
# import dill

# Compatible with tensorflow backend

# def focal_loss(gamma=2., alpha=.25):
# 	def focal_loss_fixed(y_true, y_pred):
# 		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
# 		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
# 		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
# 	return 

# def focal_loss(y_true, y_pred):
#     gamma=2.
#     alpha=.25
#     pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#     pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#     return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))


def focal(alpha=0.25, gamma=2.):
    def _focal(y_true, y_pred):
        # y_true [batch_size, num_anchor, num_classes+1]
        # y_pred [batch_size, num_anchor, num_classes]
        labels         = y_true[:, :, :-1]
        anchor_state   = y_true[:, :, -1]  # -1 是需要忽略的, 0 是背景, 1 是存在目标
        classification = y_pred

        # Find the a priori frame in which the goal exists
        indices_for_object        = tf.where(keras.backend.equal(anchor_state, 1))
        labels_for_object         = tf.gather_nd(labels, indices_for_object)
        classification_for_object = tf.gather_nd(classification, indices_for_object)

        # Calculate the weight that each a priori box should have
        alpha_factor_for_object = keras.backend.ones_like(labels_for_object) * alpha
        alpha_factor_for_object = tf.where(keras.backend.equal(labels_for_object, 1), alpha_factor_for_object, 1 - alpha_factor_for_object)
        focal_weight_for_object = tf.where(keras.backend.equal(labels_for_object, 1), 1 - classification_for_object, classification_for_object)
        focal_weight_for_object = alpha_factor_for_object * focal_weight_for_object ** gamma

        # Multiply the weights by the resulting cross-entropy
        cls_loss_for_object = focal_weight_for_object * keras.backend.binary_crossentropy(labels_for_object, classification_for_object)

        # Find the a priori box that is actually the background
        indices_for_back        = tf.where(keras.backend.equal(anchor_state, 0))
        labels_for_back         = tf.gather_nd(labels, indices_for_back)
        classification_for_back = tf.gather_nd(classification, indices_for_back)

        # Calculate the weight that each a priori box should have
        alpha_factor_for_back = keras.backend.ones_like(labels_for_back) * (1 - alpha)
        focal_weight_for_back = classification_for_back
        focal_weight_for_back = alpha_factor_for_back * focal_weight_for_back ** gamma

        # Multiply the weights by the resulting cross-entropy
        cls_loss_for_back = focal_weight_for_back * keras.backend.binary_crossentropy(labels_for_back, classification_for_back)

        # Standardised, in effect, positive sample size
        normalizer = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        # Divide the obtained loss by the number of positive samples
        cls_loss_for_object = keras.backend.sum(cls_loss_for_object)
        cls_loss_for_back = keras.backend.sum(cls_loss_for_back)

        # Total loss
        loss = (cls_loss_for_object + cls_loss_for_back)/normalizer

        return loss
    return _focal

# def binary_focal_loss(gamma=2., alpha=.25):
#     """
#     Binary form of focal loss.
#       FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
#       where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
#     References:
#         https://arxiv.org/pdf/1708.02002.pdf
#     Usage:
#      model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
#     """

#     def binary_focal_loss_fixed(y_true, y_pred):
#         """
#         :param y_true: A tensor of the same shape as `y_pred`
#         :param y_pred:  A tensor resulting from a sigmoid
#         :return: Output tensor.
#         """
#         y_true = tf.cast(y_true, tf.float32)
#         # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
#         epsilon = K.epsilon()
#         # Add the epsilon to prediction value
#         # y_pred = y_pred + epsilon
#         # Clip the prediciton value
#         y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
#         # Calculate p_t
#         p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
#         # Calculate alpha_t
#         alpha_factor = K.ones_like(y_true) * alpha
#         alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
#         # Calculate cross entropy
#         cross_entropy = -K.log(p_t)
#         weight = alpha_t * K.pow((1 - p_t), gamma)
#         # Calculate focal loss
#         loss = weight * cross_entropy
#         # Sum the losses in mini_batch
#         loss = K.mean(K.sum(loss, axis=1))
#         return loss

#     return binary_focal_loss_fixed


# if __name__ == '__main__':

#     # Test serialization of nested functions
#     bin_inner = dill.loads(dill.dumps(binary_focal_loss(gamma=2., alpha=.25)))
#     print(bin_inner)


def unet(pretrained_weights = None, input_size = (128, 128, 3), classNum = 2, learning_rate = 0.0001):
    inputs = Input(input_size)
    #  2D Convolutional Layer
    conv1 = BatchNormalization()(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs))
    conv1 = BatchNormalization()(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1))
    #  For maximum pooling of spatial data
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = BatchNormalization()(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1))
    conv2 = BatchNormalization()(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = BatchNormalization()(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2))
    conv3 = BatchNormalization()(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = BatchNormalization()(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3))
    conv4 = BatchNormalization()(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4))
    #  Dropout regularisation to prevent overfitting
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
 
    conv5 = BatchNormalization()(Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4))
    conv5 = BatchNormalization()(Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5))
    drop5 = Dropout(0.5)(conv5)
    #  Upsampling followed by convolution is equivalent to the transpose-convolution operation
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    
    try:
        merge6 = concatenate([drop4,up6],axis = 3)
    except:
        merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    conv6 = BatchNormalization()(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6))
    conv6 = BatchNormalization()(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6))
 
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    try:
        merge7 = concatenate([conv3,up7],axis = 3)
    except:
        merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = BatchNormalization()(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7))
    conv7 = BatchNormalization()(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7))
 
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    try:
        merge8 = concatenate([conv2,up8],axis = 3)
    except:
        merge8 = merge([conv2,up8],mode = 'concat', concat_axis = 3)
    conv8 = BatchNormalization()(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8))
    conv8 = BatchNormalization()(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8))
 
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    try:
        merge9 = concatenate([conv1,up9],axis = 3)
    except:
        merge9 = merge([conv1,up9],mode = 'concat', concat_axis = 3)
    conv9 = BatchNormalization()(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9))
    conv9 = BatchNormalization()(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9))
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(classNum, 1, activation = 'sigmoid')(conv9)
 
    model = Model(inputs = inputs, outputs = conv10)
 
    #  For configuring training models (optimiser, objective function, model evaluation criteria)
    #model.compile(optimizer = Adam(lr = learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy'])
    weights = {0: 0.50454072, 1: 55.55734012}
    model.compile(optimizer= Adam(lr = learning_rate), loss = focal(), metrics=['accuracy'])

    #  If there are pre-trained weights
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    
    return model

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
# from tensorflow.keras.layers import Activation, add, multiply, Lambda
# from tensorflow.keras.layers import AveragePooling2D, average, UpSampling2D, Dropout
# from tensorflow.keras.optimizers import Adam, SGD, RMSprop
# from tensorflow.keras.initializers import glorot_normal, random_normal, random_uniform
# from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D 
# from tensorflow.keras.applications import VGG19, densenet
# from tensorflow.keras.models import load_model

# import numpy as np
# import tensorflow as tf 
 
# import matplotlib.pyplot as plt 
# from sklearn.metrics import roc_curve, auc, precision_recall_curve # roc curve tools
# from sklearn.model_selection import train_test_split
# kinit = 'glorot_normal'

# def expend_as(tensor, rep,name):
# 	my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep},  name='psi_up'+name)(tensor)
# 	return my_repeat


# def AttnGatingBlock(x, g, inter_shape, name):
#     ''' take g which is the spatially smaller signal, do a conv to get the same
#     number of feature channels as x (bigger spatially)
#     do a conv on x to also get same geature channels (theta_x)
#     then, upsample g to be same size as x 
#     add x and g (concat_xg)
#     relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients'''
    
#     shape_x = K.int_shape(x)  # 32
#     shape_g = K.int_shape(g)  # 16

#     theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same', name='xl'+name)(x)  # 16
#     shape_theta_x = K.int_shape(theta_x)

#     phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
#     upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same', name='g_up'+name)(phi_g)  # 16

#     concat_xg = add([upsample_g, theta_x])
#     act_xg = Activation('relu')(concat_xg)
#     psi = Conv2D(1, (1, 1), padding='same', name='psi'+name)(act_xg)
#     sigmoid_xg = Activation('sigmoid')(psi)
#     shape_sigmoid = K.int_shape(sigmoid_xg)
#     upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

#     upsample_psi = expend_as(upsample_psi, shape_x[3],  name)
#     y = multiply([upsample_psi, x], name='q_attn'+name)

#     result = Conv2D(shape_x[3], (1, 1), padding='same',name='q_attn_conv'+name)(y)
#     result_bn = BatchNormalization(name='q_attn_bn'+name)(result)
#     return result_bn

# def UnetConv2D(input, outdim, is_batchnorm, name):
# 	x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name+'_1')(input)
# 	if is_batchnorm:
# 		x =BatchNormalization(name=name + '_1_bn')(x)
# 	x = Activation('relu',name=name + '_1_act')(x)

# 	x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name+'_2')(x)
# 	if is_batchnorm:
# 		x = BatchNormalization(name=name + '_2_bn')(x)
# 	x = Activation('relu', name=name + '_2_act')(x)
# 	return x
	

# def UnetGatingSignal(input, is_batchnorm, name):
#     ''' this is simply 1x1 convolution, bn, activation '''
#     shape = K.int_shape(input)
#     x = Conv2D(shape[3] * 1, (1, 1), strides=(1, 1), padding="same",  kernel_initializer=kinit, name=name + '_conv')(input)
#     if is_batchnorm:
#         x = BatchNormalization(name=name + '_bn')(x)
#     x = Activation('relu', name = name + '_act')(x)
#     return x

# # plain old attention gates in u-net, NO multi-input, NO deep supervision
# def attn_unet(learning_rate, input_size, lossfxn=losses.focal_tversky):   
#     inputs = Input(shape=input_size)
#     conv1 = UnetConv2D(inputs, 32, is_batchnorm=True, name='conv1')
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
#     conv2 = UnetConv2D(pool1, 32, is_batchnorm=True, name='conv2')
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

#     conv3 = UnetConv2D(pool2, 64, is_batchnorm=True, name='conv3')
#     #conv3 = Dropout(0.2,name='drop_conv3')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

#     conv4 = UnetConv2D(pool3, 64, is_batchnorm=True, name='conv4')
#     #conv4 = Dropout(0.2, name='drop_conv4')(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
#     center = UnetConv2D(pool4, 128, is_batchnorm=True, name='center')
    
#     g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
#     attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
#     up1 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], name='up1')
    
#     g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
#     attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
#     up2 = concatenate([Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up1), attn2], name='up2')

#     g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
#     attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
#     up3 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up2), attn3], name='up3')

#     up4 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up3), conv1], name='up4')
#     out = Conv2D(2, (1, 1), activation='sigmoid',  kernel_initializer=kinit, name='final')(up4)
    
#     model = Model(inputs=[inputs], outputs=[out])
#     model.compile(optimizer=Adam(lr=learning_rate), loss=lossfxn, metrics=[losses.dsc,losses.tp,losses.tn])
#     return model
