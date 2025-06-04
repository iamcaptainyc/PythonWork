from keras.layers import Lambda
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3 
from keras.applications.densenet import DenseNet121
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.layers import *
from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from matplotlib import pyplot as plt
from keras import backend as K
from keras import ops
import os
from sklearn.metrics import *
import h5py
import torch

def Global_attention_block(inputs):
    shape=inputs.shape
    x=AveragePooling2D(pool_size=(shape[1],shape[2])) (inputs)
    x=Conv2D(shape[3],1, padding='same') (x)
    x=Activation('relu') (x)
    x=Conv2D(shape[3],1, padding='same') (x)
    x=Activation('sigmoid') (x)
    C_A=Multiply()([x,inputs])
    
    x=Lambda(lambda x: ops.mean(x,axis=-1,keepdims=True))  (C_A)
    x=Activation('sigmoid') (x)
    S_A=Multiply()([x,C_A])
    return S_A
    
    
def Category_attention_block(inputs,classes,k):
    shape=inputs.shape
    F=Conv2D(k*classes,1, padding='same') (inputs)
    F=BatchNormalization() (F)
    F1=Activation('relu') (F)
    
    F2=F1
    x=GlobalMaxPool2D()(F2)
    
    x=Reshape((classes,k)) (x)
    S=Lambda(lambda x: ops.mean(x,axis=-1,keepdims=False))  (x)
    
    x=Reshape((shape[1],shape[2],classes,k)) (F1)
    x=Lambda(lambda x: ops.mean(x,axis=-1,keepdims=False))  (x)
    x=Multiply()([S,x])
    M=Lambda(lambda x: ops.mean(x,axis=-1,keepdims=True))  (x)
    
    semantic=Multiply()([inputs,M])
    return semantic 

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points    
   
def plotmodel(history,name):
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1) 
    
    plt.figure(1)                  
    plt.plot(epochs,smooth_curve(acc))
    plt.plot(epochs,smooth_curve(val_acc))
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'val_acc'], loc='upper left')
    plt.savefig('acc_'+name+'.png')
    
    plt.figure(2)
    plt.plot(epochs,smooth_curve(loss))
    plt.plot(epochs,smooth_curve(val_loss))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper right')
    plt.savefig('loss_'+name+'.png')
    
def label_smooth(y_true, y_pred):
    y_true=((1-0.1)*y_true+0.05)
    return K.categorical_crossentropy(y_true, y_pred) 
 
    
def get_base_model(model_name,image_size):
    if model_name =='vgg16':
        base_model=VGG16              (include_top=False,weights='imagenet',input_shape=(image_size,image_size,3))
    if model_name =='resnet50':
        base_model=ResNet50           (include_top=False,weights='imagenet',input_shape=(image_size,image_size,3))
    if model_name =='xception':
        base_model=Xception           (include_top=False, weights='imagenet',input_shape=(image_size,image_size,3))
    if model_name =='densenet121':
        base_model=DenseNet121       (include_top=False, weights='imagenet',input_shape=(image_size,image_size,3))
    if model_name =='mobilenet0.75':
        base_model=MobileNet         (include_top=False,weights='imagenet',alpha=0.75,input_shape=(image_size,image_size,3))
    if model_name =='mobilenet1.0':
        base_model=MobileNet         (include_top=False,weights='imagenet',alpha=1.0,input_shape=(image_size,image_size,3))
    if model_name =='mobilenetv2':
        base_model=MobileNetV2      (include_top=False,weights='imagenet',alpha=1.0,input_shape=(image_size,image_size,3))
    if model_name =='inceptionv3':   
        base_model=InceptionV3       (include_top=False,weights='imagenet',input_shape=(image_size,image_size,3))
    if model_name =='inceptionv2':
        base_model=InceptionResNetV2 (include_top=False, weights='imagenet',input_shape=(image_size,image_size,3))
    return base_model
 
    
def train_model(model,dataset,image_size,batch_size,save_name,lr1,lr2,Epochs1,Epochs2):
    
    dataParam={'messidor': [960,240,2,'/kaggle/working/messidor/train','/kaggle/working/messidor/val']} #6119
    
    train_num,valid_num,classes,train_dir,test_dir = dataParam[dataset]
    
    train=ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=90)          
    valid = ImageDataGenerator()
    train_data=train.flow_from_directory(train_dir,
                                         target_size=(image_size,image_size),
                                         shuffle = True,
                                         batch_size=batch_size)
    valid_data=valid.flow_from_directory(test_dir,
                                         target_size=(image_size,image_size),
                                         shuffle = False,
                                         batch_size=batch_size)

    lr_decay=ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1)
    model_cpt='/kaggle/working/cabnet'
    mkdir(model_cpt)
    save_model=ModelCheckpoint(model_cpt+'/'+save_name+'{epoch:02d}.keras', monitor='val_loss',save_freq=10)
    
    for layer in base_model.layers:
        layer.trainable = False   
        
    model.compile(optimizer=Adam(learning_rate=lr1,weight_decay=0.00001),loss=loss_fun,metrics=['acc'])
    model.fit(train_data,
                        steps_per_epoch=int(train_num/batch_size),
                        validation_data=valid_data,
                        validation_steps=int(valid_num/batch_size),
                        epochs=Epochs1,
                        callbacks=[lr_decay,save_model])   
    
    for layer in base_model.layers:
        layer.trainable = True
        
    model.compile(optimizer=Adam(learning_rate=lr2,weight_decay=0.00001),loss=loss_fun,metrics=['acc'])
    history=model.fit(train_data,
                        steps_per_epoch=int(train_num/batch_size),
                        validation_data=valid_data,
                        validation_steps=int(valid_num/batch_size),
                        epochs=Epochs2,
                        callbacks=[lr_decay,save_model])
    return history
      
                  
os.environ["CUDA_VISIBLE_DEVICES"] = "4"    
loss_fun= 'categorical_crossentropy'  
gpu_num=1
k=5
lr1=0.005
lr2=0.0001
batch_size= 32
image_size=512
classes=5

base_model=get_base_model('resnet50',image_size)  
base_in=base_model.input
base_out=base_model.output

x=Global_attention_block(base_out)
base_out=Category_attention_block(x,classes,k)

x=GlobalAveragePooling2D()(base_out)
out=Dense(classes,activation='softmax')(x)
model=Model(base_model.input,out)
model_name='/kaggle/input/cabnet/keras/cabnet_resnet_eyepacs/1/resnet50_CAB_EyePACS.h5'
model_path=os.path.join(model_name)
model.load_weights(model_path, by_name=True)
model.summary()
# weights=h5py.File(model_path,'r')
# for m,w in zip(model.layers,weights['model_weights'].keys()):
#     print(f'{m}---{w}')

# weights_file.close()

# test_datagen = ImageDataGenerator()  # 归一化处理
# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(image_size,image_size),
#     batch_size=batch_size,
#     class_mode='binary',  # 二分类
#     shuffle=False         # 保持顺序一致，方便后续计算
# )
                     
# y_pred_prob = model.predict(test_generator)



# # 可选：计算预测正确的比例
# accuracy = len(correct_file_names) / len(y_true)
# print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

# 提取真实标签
# y_true = test_generator.classes  # 真实标签
# y_true = y_true.reshape(-1)       # 转换为一维数组

# plot_roc(y_true, y_pred_prob[:,1])


# if gpu_num>1:
#     model=Model(base_model.input,out)
#     #model.summary()
#     parallel_model = multi_gpu_model(model, gpus=gpu_num)
#     parallel_model.summary()
# else:
#     parallel_model=Model(base_model.input,out)
#     parallel_model.summary()
    
# history=train_model(model,'messidor',image_size,batch_size,'densenet121',lr1,lr2,1,70)
# plotmodel(history,'densenet121')