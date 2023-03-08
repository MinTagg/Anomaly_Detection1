import json
import numpy as np
import tensorflow as tf
import pandas as pd
import train_argument as arg
import os
import cv2
from skimage.transform import resize
import random


class ano_dataset(tf.data.Dataset):
    """
    This dataloader is used in self-supervised learning that has 3 tasks(Arrow of Time, Motion Irregularity, Predict Middle Frame)
    Thod dataloader don't crop image and oprical flow data. It will serve the whole 5 frames scene 
    """
    def _generator():
        margin = (arg.model_input-1)//2
        df = pd.read_csv(arg.data_csv_path)
        df = df.sample(frac=1) # 데이터 프레임 셔플
        for _, row in df.iterrows(): # csv 파일에서 하나의 줄을 읽어온다
            folder, name = row['Path'].split('/') # name = 1550.jpg, folder = 1
            mid_index = arg.model_input - margin - 1
            state = True
            image_file_name = []
            flow_file_name = []
            for i in range(arg.model_input): # 열기 위한 파일의 이름들을 저장한다
            #    print(f'Image File check :: {arg.image_path}{folder}/{str(int(name[:-4])-margin+i)}.jpg')
            #    print(f'Flow File check :: {arg.flow_path}{folder}/{str(int(name[:-4])-margin+i)}.flo')
                
                image_file_name.append(f'{arg.image_path}{folder}/{str(int(name[:-4])-margin+i)}.jpg')
                flow_file_name.append(f'{arg.flow_path}{folder}/{str(int(name[:-4])-margin+i)}.flo')
                
                # json_file_name = 'data/json/1/1543.json
                if os.path.isfile(image_file_name[-1]) == False: # 해당 파일이 존재하지 않는다면 state를 False, 다음 csv 파일을 읽는다
                    state = False
                    break
                if os.path.isfile(flow_file_name[-1]) == False: # 해당 파일이 존재하지 않는다면 state를 False, 다음 csv 파일을 읽는다
                    state = False
                    break
            if state: # 모든 이미지, flow 파일이 존재한다면
                concated_list = []
                for index in range(len(image_file_name)):
                    image_path = image_file_name[index]
                    flow_path = flow_file_name[index]
                    
                    image = cv2.imread(image_path)
                    image = resize(image, arg.flow_size)
                    image = np.expand_dims(image, axis = 0)

                    with open(flow_path, 'rb') as f:
                        _ = np.fromfile(f, np.float32, count=1)
                        w = np.fromfile(f, np.int32, count=1)
                        h = np.fromfile(f, np.int32, count=1)
                        flow = np.fromfile(f, np.float32, count= 2*int(w) * int(h))
                        flow = np.resize(flow, (int(h), int(w), 2))
                    
                    flow = resize(flow, arg.flow_size)
                    flow = np.expand_dims(flow, axis = 0)

                    image = np.concatenate((image, flow), axis = -1)
                    concated_list.append(image)
                ### 전체 이미지를 resize 해서 concated_list에 저장

                aot_index = [i for i in range(arg.model_input)]
                aot_concated = []
                aot_label = [1,0]

                probability = random.random()
                if probability < arg.prob:
                #    print('Arrow of Time')
                    aot_index.reverse()
                    aot_label = [0,1]
                
                for index in aot_index:
                    if len(aot_concated) == 0:
                        aot_concated = concated_list[index]
                    else:
                        aot_concated = np.concatenate((aot_concated, concated_list[index]), axis = 0)


                moi_index = [i for i in range(arg.model_input)]
                moi_concated = []
                moi_label = [1,0]

                probability = random.random()
                if probability < arg.prob:
                    #print('Motion Irregularity')
                    temp = random.random()
                    if temp >= 0 and temp >= 0.25:
                        moi_index[1] = moi_index[0] # t-2, t-2, t ,t+1, t+2
                    #    print('t-2, t-2, t ,t+1, t+2')
                    elif temp >= 0.5:
                        moi_index[0] = moi_index[1] # t-1, t-1, t, t+1, t+2
                    #    print('t-1, t-1, t, t+1, t+2')
                    elif temp >= 0.75:
                        moi_index[3] = moi_index[4] # t-2, t-1, t, t+2, t+2
                    #    print('t-2, t-1, t, t+2, t+2')
                    else:
                        moi_index[4] = moi_index[4] # t-2, t-1, t, t+1, t+1
                    #    print('t-2, t-1, t, t+1, t+1')
                    moi_label = [0,1]

                for index in moi_index:
                    if len(moi_concated) == 0:
                        moi_concated = concated_list[index]
                    else:
                        moi_concated = np.concatenate((moi_concated, concated_list[index]), axis = 0)
                    
                for i in range(arg.model_input):
                    if i == 0:
                        mp_concated = concated_list[i]
                    elif i == mid_index:
                        mp_concated = np.concatenate((mp_concated, np.zeros(shape = (1, arg.flow_size[0], arg.flow_size[1], 5))), axis = 0)
                    else:
                        mp_concated = np.concatenate((mp_concated, concated_list[i]), axis = 0)
                mp_label = np.reshape(concated_list[mid_index], (arg.flow_size[0], arg.flow_size[1], -1))


                yield aot_concated, aot_label, moi_concated, moi_label, mp_concated, mp_label, folder+'/'+name


                    



    def __new__(cls):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape = (arg.model_input, arg.flow_size[0], arg.flow_size[1], 5), dtype = tf.float32),
                tf.TensorSpec(shape=(2,), dtype=tf.float32),

                tf.TensorSpec(shape = (arg.model_input, arg.flow_size[0], arg.flow_size[1], 5), dtype = tf.float32),
                tf.TensorSpec(shape=(2,), dtype=tf.float32),

                tf.TensorSpec(shape = (arg.model_input, arg.flow_size[0], arg.flow_size[1], 5), dtype = tf.float32),
                tf.TensorSpec(shape = (arg.flow_size[0], arg.flow_size[1], 5), dtype = tf.float32),

                tf.TensorSpec(shape=(), dtype= tf.string),
            )
        )
