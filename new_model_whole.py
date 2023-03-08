import tensorflow as tf
import train_argument as arg
"""
Tensorflow model that use whole scene of data
"""
def block(before, name_tag, epoch=1, kernel = 3, filter = None, ):
    temp = before
    if filter == None: # 이미지 필터 크기를 바꾸지 않는다면 
        filter = before.shape[-1]
    for i in range(epoch):
        x = tf.keras.layers.Conv3D(filters = filter, kernel_size= kernel, padding = 'same', activation= None, name = f'{name_tag}_{i+1}_1_Conv')(temp)
        x = tf.keras.layers.BatchNormalization(name = f'{name_tag}_{i+1}_1_BatNorm')(x)
        x = tf.keras.layers.ReLU(name = f'{name_tag}_{i+1}_1_ReLU')(x)
        x = tf.keras.layers.Conv3D(filters = filter, kernel_size= kernel, padding = 'same', activation= None,name = f'{name_tag}_{i+1}_2_Conv')(x)
        x = tf.keras.layers.BatchNormalization(name = f'{name_tag}_{i+1}_2_BatNorm')(x)
        if (temp.shape[-1]) != filter: # 필터의 수가 바뀌는 경우
            temp = tf.keras.layers.Conv3D(filters = filter, kernel_size=1, padding = 'same', activation= None, name = f'{name_tag}_{i+1}_residual_filter')(temp)
            temp = tf.keras.layers.BatchNormalization(name = f'{name_tag}_{i+1}_residual_BatNorm')(temp)
        x = tf.keras.layers.Add(name = f'{name_tag}_{i+1}_Add')([temp, x])
        temp = tf.keras.layers.ReLU(name = f'{name_tag}_{i+1}_2_ReLU')(x)
    return temp

def front():
    # 모델 입력 
    inputs = tf.keras.layers.Input(shape=(arg.model_input, arg.flow_size[0], arg.flow_size[1], 5), name = 'Input')
    # 기본 필터 = 5
    filters = arg.front_filters
    epochs =  arg.front_epochs
    pools = arg.front_pools
    b = inputs
    if len(filters) != len(epochs) != len(pools):
        print('Error Occured :: Filter, Epoch, Pooling size Not matched')
        return None

    for i in range(len(filters)):
        if pools[i]:
            b = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides = (1,2,2), name = f'{i+1}_MaxPooling')(b)
        b = block(b, name_tag= f'{i+1}', filter = filters[i], epoch = epochs[i])

    #b1 = block(inputs,name_tag = '1', epoch = 2)
    #b2 = block(b1, name_tag = '2', filter = 10, epoch = 2)
    #b3 = block(b2, name_tag= '3', filter = 20, epoch = 3)
    model = tf.keras.Model(inputs = inputs, outputs = b, name = 'Front')
    
    return model

def back():
    size = (arg.flow_size[0]//(2**arg.front_pools.count(True)),arg.flow_size[1]//(2**arg.front_pools.count(True)))
    inputs = tf.keras.layers.Input(shape = (arg.model_input, size[0], size[1], arg.front_filters[-1]), name = 'Rear Input')

    #back_sequence_pools = arg.back_sequence_pools # 5 -> 5 -> 4-> 3-> 2-> 1
    #back_channel_pools = arg.back_channel_pools # 64 -> 32 -> 16 -> 8 -> 4 -> 2
    #back_epochs = arg.back_epochs
    #back_filters = arg.back_filters         # 512 -> 512 -> 1024 -> 1024 -> 2048 -> 2048

    back_sequence_pools = arg.back_sequence_pools # 5 -> 5 -> 4-> 3-> 2-> 1
    back_channel_pools = arg.back_channel_pools # 64 -> 32 -> 16 -> 8 -> 4 -> 2
    back_epochs = arg.back_epochs
    back_filters = arg.back_filters       # 512 -> 512 -> 1024 -> 1024 -> 2048 -> 2048


    b = inputs
    if len(back_sequence_pools) != len(back_channel_pools) != len(back_epochs) != len(back_filters):
        print('Error Occured :: Filter, Epoch, Pooling size Not matched')
        return None

    for i in range(len(back_filters)):
        if back_channel_pools[i]:
            b = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides = (1,2,2), name = f'Back_{i+1}_Channel_pooling')(b)
        if back_sequence_pools[i]:
            b = tf.keras.layers.MaxPool3D(pool_size = (2,1,1), strides = (1,1,1), name = f'Back_{i+1}_Sequence_pooling')(b)
        b = block(b, name_tag=f'Back_{i+1}', filter = back_filters[i], epoch = back_epochs[i])
    
    model = tf.keras.Model(inputs = inputs, outputs = b, name = 'Rear_3D_Conv')

    return model

def arrow_of_time(name = None):
    inputs = tf.keras.layers.Input(shape = (arg.flow_size[0]//(2**(arg.front_pools.count(True)+arg.back_channel_pools.count(True))),arg.flow_size[0]//(2**(arg.front_pools.count(True)+arg.back_channel_pools.count(True))), arg.back_filters[-1]))
    # shape = 2, 2, 2048 -> 2, 2, 1024 -> 2, 2, 512 -> 2, 2, 256 -> 256 -> 2

    aot_filters = arg.aot_filters
    aot_pooling = arg.aot_pooling

    if len(aot_filters) != len(aot_pooling):
        print('Error Occur :: AOT Filter/Pooling size not Matched')
        return None

    x = inputs
    for index in range(len(aot_filters)):
        x = tf.keras.layers.Conv2D(filters = aot_filters[index] , kernel_size=3, padding = 'same', activation='relu', name = f'arrow_of_time_{index+1}_conv2d')(x)
        if aot_pooling[index]:
            x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name = f'arrow_of_time_{index+1}_pooling')(x)

    x = tf.keras.layers.Flatten(name = 'arrow_of_time_flatten')(x)
    x = tf.keras.layers.Dense(2, activation = 'softmax')(x)

    model = tf.keras.Model(inputs = inputs, outputs = x, name = name)

    return model

def middle_prediction():
    inputs = tf.keras.layers.Input(shape = (arg.flow_size[0]//(2**(arg.front_pools.count(True)+arg.back_channel_pools.count(True))),arg.flow_size[0]//(2**(arg.front_pools.count(True)+arg.back_channel_pools.count(True))), arg.back_filters[-1]))
    # 2, 2, 2048 -> 4, 4, 1024 -> 8, 8, 512 -> 16, 16, 128 -> 32, 32, 64 -> 64, 64, 32 -> 128, 128, 8 -> 256, 256, 6
    # 목표 = 256, 256, 6
    mp_filters = arg.mp_filters
    mp_upsample = arg.mp_upsample

    if len(mp_filters) != len(mp_upsample):
        print('Error Occured :: Middle Prediction length not Matched')
        return None

    x = inputs
    for index in range(len(mp_filters)):
        x = tf.keras.layers.Conv2D(filters = mp_filters[index], kernel_size = 3, padding = 'same', activation = None, name = f'Middle_prediction_{index+1}_Conv2d')(x)
        x = tf.keras.layers.BatchNormalization(name = f'Middle_prediction_{index+1}_BatchNorm')(x)
        x = tf.keras.layers.ReLU(name = f'Middle_prediction_{index+1}_ReLU')(x)
        if mp_upsample[index]:
            x = tf.keras.layers.UpSampling2D(size = (2,2), name = f'Middle_prediction_{index+1}_Upsampling')(x)
    
    model = tf.keras.Model(inputs = inputs, outputs = x, name = 'Middle_Prediction')

    return model

        
def model():
    """
    Model that use whole scene and 3D Convolution
    """
    AOT_front_input = tf.keras.layers.Input(shape=(arg.model_input, arg.flow_size[0], arg.flow_size[1], 5), name = 'AOT_Front_Input')
    
    MOI_front_input = tf.keras.layers.Input(shape=(arg.model_input, arg.flow_size[0], arg.flow_size[1], 5), name = 'MOI_Front_Input')

    MP_front_input = tf.keras.layers.Input(shape=(arg.model_input, arg.flow_size[0], arg.flow_size[1], 5), name = 'MP_Front_Input')

    # Encoding Model
    front_model = front()
    back_model = back()

    # Task model
    aot = arrow_of_time('Arrow_of_Time')
    motion = arrow_of_time('Motion_Irregularity')
    mid_pre = middle_prediction()

    # front
    AOT_front_result = front_model(AOT_front_input)
    #AOT_mlp_result = tf.keras.layers.Reshape(target_shape=(-1,1,1,1))(AOT_mlp_result)

    MOI_front_result = front_model(MOI_front_input)
    #MOI_mlp_result = tf.keras.layers.Reshape(target_shape=(-1,1,1,1))(MOI_mlp_result)

    MP_front_result = front_model(MP_front_input)
    #MP_mlp_result = tf.keras.layers.Reshape(target_shape=(-1,1,1,1))(MP_mlp_result)

    # front fusion
    AOT_ENCODED = back_model(AOT_front_result)
    AOT_ENCODED = tf.keras.layers.Reshape(target_shape = (arg.flow_size[0]//(2**(arg.front_pools.count(True)+arg.back_channel_pools.count(True))),arg.flow_size[0]//(2**(arg.front_pools.count(True)+arg.back_channel_pools.count(True))), -1))(AOT_ENCODED)

    MOI_ENCODED = back_model(MOI_front_result)
    MOI_ENCODED = tf.keras.layers.Reshape(target_shape = (arg.flow_size[0]//(2**(arg.front_pools.count(True)+arg.back_channel_pools.count(True))),arg.flow_size[0]//(2**(arg.front_pools.count(True)+arg.back_channel_pools.count(True))), -1))(MOI_ENCODED)

    MP_ENCODED = back_model(MP_front_result)
    MP_ENCODED = tf.keras.layers.Reshape(target_shape = (arg.flow_size[0]//(2**(arg.front_pools.count(True)+arg.back_channel_pools.count(True))),arg.flow_size[0]//(2**(arg.front_pools.count(True)+arg.back_channel_pools.count(True))), -1))(MP_ENCODED)

    # Task
    aot_result = aot(AOT_ENCODED)
    motion_result = motion(MOI_ENCODED)
    mid_pre_result = mid_pre(MP_ENCODED)
    ### 이후 3D Convolution 붙이기

    model = tf.keras.Model(inputs = [AOT_front_input, MOI_front_input, MP_front_input], outputs = [aot_result, motion_result, mid_pre_result], name = '3DNN')

    return model

def MP_model():
    # Inputs
    AOT_front_input = tf.keras.layers.Input(shape=(arg.model_input, arg.flow_size[0], arg.flow_size[1], 5), name = 'AOT_Front_Input')
    
    MOI_front_input = tf.keras.layers.Input(shape=(arg.model_input, arg.flow_size[0], arg.flow_size[1], 5), name = 'MOI_Front_Input')

    MP_front_input = tf.keras.layers.Input(shape=(arg.model_input, arg.flow_size[0], arg.flow_size[1], 5), name = 'MP_Front_Input')

    # Encoding Model
    front_model = front()
    back_model = back()

    # Task model
    aot = arrow_of_time('Arrow_of_Time')
    aot.trainable = False
    motion = arrow_of_time('Motion_Irregularity')
    motion.trainable = False
    mid_pre = middle_prediction()


    # front
    AOT_front_result = front_model(AOT_front_input)
    #AOT_mlp_result = tf.keras.layers.Reshape(target_shape=(-1,1,1,1))(AOT_mlp_result)

    MOI_front_result = front_model(MOI_front_input)
    #MOI_mlp_result = tf.keras.layers.Reshape(target_shape=(-1,1,1,1))(MOI_mlp_result)

    MP_front_result = front_model(MP_front_input)
    #MP_mlp_result = tf.keras.layers.Reshape(target_shape=(-1,1,1,1))(MP_mlp_result)

    # front fusion
    AOT_ENCODED = back_model(AOT_front_result)
    AOT_ENCODED = tf.keras.layers.Reshape(target_shape = (arg.flow_size[0]//(2**(arg.front_pools.count(True)+arg.back_channel_pools.count(True))),arg.flow_size[0]//(2**(arg.front_pools.count(True)+arg.back_channel_pools.count(True))), -1))(AOT_ENCODED)

    MOI_ENCODED = back_model(MOI_front_result)
    MOI_ENCODED = tf.keras.layers.Reshape(target_shape = (arg.flow_size[0]//(2**(arg.front_pools.count(True)+arg.back_channel_pools.count(True))),arg.flow_size[0]//(2**(arg.front_pools.count(True)+arg.back_channel_pools.count(True))), -1))(MOI_ENCODED)

    MP_ENCODED = back_model(MP_front_result)
    MP_ENCODED = tf.keras.layers.Reshape(target_shape = (arg.flow_size[0]//(2**(arg.front_pools.count(True)+arg.back_channel_pools.count(True))),arg.flow_size[0]//(2**(arg.front_pools.count(True)+arg.back_channel_pools.count(True))), -1))(MP_ENCODED)

    # Task
    aot_result = aot(AOT_ENCODED)
    motion_result = motion(MOI_ENCODED)
    mid_pre_result = mid_pre(MP_ENCODED)
    ### 이후 3D Convolution 붙이기

    model = tf.keras.Model(inputs = [AOT_front_input, MOI_front_input, MP_front_input], outputs = [aot_result, motion_result, mid_pre_result], name = '3DNN')

    return model
