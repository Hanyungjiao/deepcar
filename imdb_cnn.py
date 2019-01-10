from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPool1D
from keras.datasets import imdb

#set parameters
max_features = 5000  # 最大特征
maxlen = 400
batch_size = 32  # 填充样本的个数
embedding_dims = 50  # 嵌入
filters = 250  # 过滤器个数
kernel_size = 3  # 卷积和的大小
hidden_dims = 250  # 隐藏 dim
epochs = 2

print("Loading data")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train_sequence')


x_train = sequence.pad_sequences(x_train, maxlen=maxlen)  # 将序列填充到指定长度
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
# 嵌入层 将正整数转换为具有固定大小的向量
model.add(Embedding(max_features,  # 字典长度
                    embedding_dims,  # 代表全连接嵌入的维度
                    input_length=maxlen))  # 当输入序列的长度固定时，该值为其长度

# Dropout   一定几率停止该隐藏神经元，防止过拟合
model.add(Dropout=0.2)

model.add(Conv1D(filters=filters,  #
                 kernel_size=kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# 最大池化
model.add(GlobalMaxPool1D())

# 隐藏层
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# 输出层和激活函数
model.add(Dense(1))  # output_dim:训练终止时的维度
model.add(Activation('sigmoid'))

# 配置模型
model.compile(loss='binary_crossentropy',  # 目标函数（损失函数）
              optimizer='adam',  # 选择优化器：随机梯度下降法， 支持动量参数， 支持学习衰减率， 支持Netsterovd动量
              metrics=['accuracy'])  # 评价方法

# 训练并预测
model.fit(x_train, y_train,
          batch_size=batch_size,  # 填充样本的个数
          epochs=epochs,  # 训练终止时的epoch值
          validation_data=(x_test, y_test))









