import tensorflow as tf
import train_argument as args

# 입력 :: image == None, Height, Width, Channel
# 계산 :: image == None

# Convolution - Batch Normalization - Activation - Dropout - Pooling 순서대로 네트워크 구성하기
"""
Model that use cropped data
It has just model, wide, deep, wide and deep
"""

def decoder_middle_frame(input_shape):

    inputs = tf.keras.Input(input_shape)

    upsample1 = tf.keras.layers.Resizing(height = input_shape[0]*2, width = input_shape[1] * 2, interpolation='nearest')(inputs)

    conv5 = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation=None)(upsample1)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = tf.keras.layers.ReLU()(conv5)


    upsample2 = tf.keras.layers.Resizing(height = conv5.shape[1]*2, width = conv5.shape[2] * 2, interpolation='nearest')(conv5)

    conv6 = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation=None)(upsample2)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    conv6 = tf.keras.layers.ReLU()(conv6)

    upsample3 = tf.keras.layers.Resizing(height = conv6.shape[1]*2, width = conv6.shape[2] * 2, interpolation='nearest')(conv6)

    conv7 = tf.keras.layers.Conv2D(filters = 3, kernel_size = (3,3), padding = 'same', activation=None)(upsample3)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = tf.keras.layers.ReLU()(conv7)

    model = tf.keras.Model(inputs, conv7, name = 'decoder_middle_frame')
    return model


def decoder_middle_frame_deep(input_shape):

    inputs = tf.keras.Input(input_shape)

    x = tf.keras.layers.Resizing(height = input_shape[0]*2, width = input_shape[1] * 2, interpolation='nearest')(inputs) # upsample

    for _ in range(3):
        x = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Resizing(height=x.shape[1]*2, width = x.shape[2]*2, interpolation='nearest')(x) # Upsanple

    x = tf.keras.layers.Conv2D(filters = 3, kernel_size = (3,3), padding = 'same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    model = tf.keras.Model(inputs, x, name = 'decoder_middle_frame_deep')

    return model

def decoder_fwd_bwd(input_shape):
    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2, activation=None)(x)

    model = tf.keras.Model(inputs, x, name = 'decoder_fwd_bwd')
    return model

def decoder_consecutive(input_shape):
    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2, activation=None)(x)

    model = tf.keras.Model(inputs, x, name = 'decoder_consecutive')
    return model

def decoder_resnet(input_shape):
    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1080, activation=None)(x)

    model = tf.keras.Model(inputs, x, name = 'decoder_resnet')
    return model

def decoder_c3d(input_shape):
    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(487, activation=None)(x)

    model = tf.keras.Model(inputs, x, name = 'decoder_c3d')
    return model

def encoder(input_shape):
    inputs = tf.keras.Input(input_shape)
    # 입력 = None, 5, 64, 64, 3 == Batch, Time, Height, Width, Channel
    conv1 = tf.keras.layers.Conv3D(filters=16, kernel_size=(3,3,3), padding = 'same', activation = None)(inputs)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.ReLU()(conv1)
    maxpool1 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv1)

    conv2 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), padding = 'same', activation = None)(maxpool1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.ReLU()(conv2)
    maxpool2 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv2)

    conv3 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), padding = 'same', activation = None)(maxpool2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.ReLU()(conv3)
    temporal = tf.keras.layers.MaxPool3D(pool_size=(conv3.shape[1],2,2), strides=2)(conv3) # temporal_max_pooling

    encoded = tf.keras.layers.Reshape(target_shape=(8,8,32))(temporal)

    model = tf.keras.Model(inputs, encoded, name = 'encoder')

    return model

def encoder_deep(input_shape):
    inputs = tf.keras.Input(input_shape)

    conv_1 = tf.keras.layers.Conv3D(filters=16, kernel_size=(3,3,3), padding = 'same', activation=None)(inputs)
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_1 = tf.keras.layers.ReLU()(conv_1)
    conv_1_1 = tf.keras.layers.Conv3D(filters=16, kernel_size=(3,3,3), padding='same', activation=None)(conv_1)
    conv_1_1 = tf.keras.layers.BatchNormalization()(conv_1_1)
    conv_1_1 = tf.keras.layers.ReLU()(conv_1_1)

    max_pool_1 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv_1_1)

    conv_2 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), padding = 'same', activation=None)(max_pool_1)
    conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    conv_2 = tf.keras.layers.ReLU()(conv_2)
    conv_2_1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), padding='same', activation=None)(conv_2)
    conv_2_1 = tf.keras.layers.BatchNormalization()(conv_2_1)
    conv_2_1 = tf.keras.layers.ReLU()(conv_2_1)

    max_pool_2 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv_2_1)

    conv_3 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), padding = 'same', activation=None)(max_pool_2)
    conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    conv_3 = tf.keras.layers.ReLU()(conv_3)

    max_pool_3 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv_3)

    conv_4 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), padding = 'same', activation=None)(max_pool_3)
    conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
    conv_4 = tf.keras.layers.ReLU()(conv_4)

    temporal_max_pooling = tf.keras.layers.MaxPool3D(pool_size=(conv_4.shape[1],2,2), strides=2)(conv_4)

    encoded = tf.keras.layers.Reshape(target_shape=(4,4,32))(temporal_max_pooling)

    model = tf.keras.Model(inputs, encoded, name = 'encoder_deep')
    return model

def encoder_deep_wide(input_shape):
    inputs = tf.keras.Input(input_shape)

    conv_1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), padding = 'same', activation=None)(inputs)
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_1 = tf.keras.layers.ReLU()(conv_1)
    conv_1_1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), padding='same', activation=None)(conv_1)
    conv_1_1 = tf.keras.layers.BatchNormalization()(conv_1_1)
    conv_1_1 = tf.keras.layers.ReLU()(conv_1_1)

    max_pool_1 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv_1_1)

    conv_2 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), padding = 'same', activation=None)(max_pool_1)
    conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    conv_2 = tf.keras.layers.ReLU()(conv_2)
    conv_2_1 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), padding='same', activation=None)(conv_2)
    conv_2_1 = tf.keras.layers.BatchNormalization()(conv_2_1)
    conv_2_1 = tf.keras.layers.ReLU()(conv_2_1)

    max_pool_2 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv_2_1)

    conv_3 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), padding = 'same', activation=None)(max_pool_2)
    conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    conv_3 = tf.keras.layers.ReLU()(conv_3)

    max_pool_3 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv_3)

    conv_4 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), padding = 'same', activation=None)(max_pool_3)
    conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
    conv_4 = tf.keras.layers.ReLU()(conv_4)

    temporal_max_pooling = tf.keras.layers.MaxPool3D(pool_size=(conv_4.shape[1],2,2), strides=2)(conv_4)

    encoded = tf.keras.layers.Reshape(target_shape=(4,4,64))(temporal_max_pooling)

    model = tf.keras.Model(inputs, encoded, name = 'encoder_deep_wide')
    return model

def encoder_wide(input_shape):
    inputs = tf.keras.Input(input_shape)
    # 입력 = None, 5, 64, 64, 3 == Batch, Time, Height, Width, Channel
    conv1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), padding = 'same', activation = None)(inputs)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.ReLU()(conv1)
    maxpool1 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv1)

    conv2 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), padding = 'same', activation = None)(maxpool1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.ReLU()(conv2)
    maxpool2 = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv2)

    conv3 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), padding = 'same', activation = None)(maxpool2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.ReLU()(conv3)
    temporal = tf.keras.layers.MaxPool3D(pool_size=(conv3.shape[1],2,2), strides=2)(conv3) # temporal_max_pooling

    encoded = tf.keras.layers.Reshape(target_shape=(8,8,64))(temporal)

    model = tf.keras.Model(inputs, encoded, name = 'encoder_wide')

    return model

# input = arrpw of time. middle box preddiction
# input_conscutive = motion irregularity
# input_resnet = model_distillation
# om[it cpmsecitove - motion irregi;aroty]
# input resnet = model distillation
def model():
    """
    Model that take 3 Task input and decoding
    It planned to train all task at once
    """
    inputs_shape = [args.model_input]
    inputs_consecutive_shape = [args.model_consecutive]
    inputs_resnet_shape = [args.model_resnet]

    #inputs_shape = [None]
    #inputs_consecutive_shape = [None]
    #inputs_resnet_shape = [None]
    for i in args.input_size:
        inputs_shape.append(i)
        inputs_consecutive_shape.append(i)
        inputs_resnet_shape.append(i)
    
    # 입력 모양 3개
    inputs_ = tf.keras.Input(inputs_shape, name = 'Arrow of time, Middle box prediction')
    inputs_consecutive = tf.keras.Input(inputs_consecutive_shape, name = 'Motion irregularity')
    inputs_resnet = tf.keras.Input(inputs_resnet_shape, name = 'Model distillation')

    # 인코더 블럭 생성
    ENCODER = encoder(inputs_shape)

    # 블럭 이용해서 값들 인코딩

    encoded = ENCODER(inputs_)
    decoder_middle_ = decoder_middle_frame(encoded.shape[1:])(encoded)
    logits_fwd_bwd = decoder_fwd_bwd(encoded.shape[1:])(encoded)

    encoded_consecutive = ENCODER(inputs_consecutive)
    logits_consecutive = decoder_consecutive(encoded_consecutive.shape[1:])(encoded_consecutive)

    encoded_resnet = ENCODER(inputs_resnet)
    logits_resnet = decoder_resnet(encoded_resnet.shape[1:])(encoded_resnet)

    result = tf.keras.Model(inputs = [inputs_, inputs_consecutive, inputs_resnet], outputs = [decoder_middle_, logits_fwd_bwd, logits_consecutive, logits_resnet])
    
    return result

def model_wide():
    inputs_shape = [args.model_input]
    inputs_consecutive_shape = [args.model_consecutive]
    inputs_resnet_shape = [args.model_resnet]
    for i in args.input_size:
        inputs_shape.append(i)
        inputs_consecutive_shape.append(i)
        inputs_resnet_shape.append(i)
    
    # 입력 모양 3개
    inputs_ = tf.keras.Input(inputs_shape)
    inputs_consecutive = tf.keras.Input(inputs_consecutive_shape)
    inputs_resnet = tf.keras.Input(inputs_resnet_shape)

    # 인코더 블럭 생성
    ENCODER = encoder_wide(inputs_shape)

    # 블럭 이용해서 값들 인코딩

    encoded = ENCODER(inputs_)
    decoder_middle_ = decoder_middle_frame(encoded.shape[1:])(encoded)
    logits_fwd_bwd = decoder_fwd_bwd(encoded.shape[1:])(encoded)

    encoded_consecutive = ENCODER(inputs_consecutive)
    logits_consecutive = decoder_consecutive(encoded_consecutive.shape[1:])(encoded_consecutive)

    encoded_resnet = ENCODER(inputs_resnet)
    logits_resnet = decoder_resnet(encoded_resnet.shape[1:])(encoded_resnet)

    result = tf.keras.Model(inputs = [inputs_, inputs_consecutive, inputs_resnet], outputs = [decoder_middle_, logits_fwd_bwd, logits_consecutive, logits_resnet])
    
    return result

def model_deep():
    inputs_shape = [args.model_input]
    inputs_consecutive_shape = [args.model_consecutive]
    inputs_resnet_shape = [args.model_resnet]
    for i in args.input_size:
        inputs_shape.append(i)
        inputs_consecutive_shape.append(i)
        inputs_resnet_shape.append(i)
    
    # 입력 모양 3개
    inputs_ = tf.keras.Input(inputs_shape)
    inputs_consecutive = tf.keras.Input(inputs_consecutive_shape)
    inputs_resnet = tf.keras.Input(inputs_resnet_shape)

    # 인코더 블럭 생성
    ENCODER = encoder_deep(inputs_shape)

    # 블럭 이용해서 값들 인코딩

    encoded = ENCODER(inputs_)
    decoder_middle_ = decoder_middle_frame(encoded.shape[1:])(encoded)
    logits_fwd_bwd = decoder_fwd_bwd(encoded.shape[1:])(encoded)

    encoded_consecutive = ENCODER(inputs_consecutive)
    logits_consecutive = decoder_consecutive(encoded_consecutive.shape[1:])(encoded_consecutive)

    encoded_resnet = ENCODER(inputs_resnet)
    logits_resnet = decoder_resnet(encoded_resnet.shape[1:])(encoded_resnet)

    result = tf.keras.Model(inputs = [inputs_, inputs_consecutive, inputs_resnet], outputs = [decoder_middle_, logits_fwd_bwd, logits_consecutive, logits_resnet])
    
    return result

def model_deep_wide():
    #inputs_shape = [args.model_input]
    #inputs_consecutive_shape = [args.model_consecutive]
    #inputs_resnet_shape = [args.model_resnet]

    inputs_shape = [args.model_input]
    inputs_consecutive_shape = [args.model_consecutive]
    inputs_resnet_shape = [args.model_resnet]
    for i in args.input_size:
        inputs_shape.append(i)
        inputs_consecutive_shape.append(i)
        inputs_resnet_shape.append(i)
    
    # 입력 모양 3개
    inputs_ = tf.keras.Input(inputs_shape)
    inputs_consecutive = tf.keras.Input(inputs_consecutive_shape)
    inputs_resnet = tf.keras.Input(inputs_resnet_shape)

    # 인코더 블럭 생성
    ENCODER = encoder_deep_wide(inputs_shape)

    # 블럭 이용해서 값들 인코딩

    encoded = ENCODER(inputs_)
    decoder_middle_ = decoder_middle_frame(encoded.shape[1:])(encoded)
    logits_fwd_bwd = decoder_fwd_bwd(encoded.shape[1:])(encoded)

    encoded_consecutive = ENCODER(inputs_consecutive)
    logits_consecutive = decoder_consecutive(encoded_consecutive.shape[1:])(encoded_consecutive)

    encoded_resnet = ENCODER(inputs_resnet)
    logits_resnet = decoder_resnet(encoded_resnet.shape[1:])(encoded_resnet)

    result = tf.keras.Model(inputs = [inputs_, inputs_consecutive, inputs_resnet], outputs = [decoder_middle_, logits_fwd_bwd, logits_consecutive, logits_resnet])
    
    return result

if __name__ == '__main__':
    a = model()
    a.summary()
    a = model_wide()
    a.summary()
    a = model_deep()
    a.summary()
    a = model_deep_wide()
    a.summary()