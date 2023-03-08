from datetime import datetime
import tensorflow as tf


input_size = [64,64,3]

learning_rate_init = 10 ** -3
num_epochs = 100
batch_size = 64

folder = 'result/'+ datetime.today().strftime(f'%Y_%m_%d_%H_%M_%S_lr_{learning_rate_init}_ep_{num_epochs}_batch_{batch_size}')

# model_four_tasks
model_input = 5
model_consecutive = 5
model_resnet = 5

# pipeline
#data_csv_path = 'data/image_path.csv'
data_csv_path = 'data/image_path.csv'

#json_path = 'data/json/'
json_path = 'data/json/'
#resnet_path = 'data/resnet/'
object_count = 10 * 4 # x,y,w,h 때문에 4를 곱해준다
num_samples = 3
resnet_output = (7,7,2048)

# Optical Flow
flow_size = (1280, 720, 2)
flow_path = 'data/flow/'
flow_size = (256,256)

# image
image_path = 'data/image/'

# class filter
# 0: 사람, 1: 자전거, 2: 차, 3: 오토바이, 5: 버스, 7:트럭
class_filter = [0,1,2,3,5,7]

# anomaly probability
prob = 0.5 # 30% 확률로 이상 데이터를 생성한다
# anomaly task list
# task 3번 middle box prediction은 변형이 불가능한 task임
task_list = [1,2]

# MODEL
front_filters = [6,8,16,32,64,128,256,512]
front_epochs = [1,1,1,2,1,1,1,2]
front_pools = [False, False, False, True, False, False, False, True]

#REAR_
back_sequence_pools = [True, True, True, True, False, False] # 5 -> 5 -> 4-> 3-> 2-> 1
back_channel_pools = [True, True, True, True, True, False] # 64 -> 32 -> 16 -> 8 -> 4 -> 2
back_epochs = [1, 2, 2, 2, 2, 1]
back_filters = [256, 128, 128, 256, 512, 1024]  


# arrow of time(aot)
aot_filters = [1024, 512, 256]
aot_pooling = [False, False, True]


# Middle Prediction
mp_filters = [1024, 512, 128, 64, 32, 8, 5]
mp_upsample = [True, True,True,True,True,True,True]


### TRAIN
batch = 4
data_size = 1000
epoch = 5