#coding:utf-8
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# Dimensões das imagens..
img_largura, img_altura = 150, 150

# Diretório para treino.
pasta_treino = 'dataset/train'

# Diretório pra validação.
pasta_validacao = 'dataset/validation'

# Samples de treinamento e validação.
samples_de_treinamento = 2000
samples_de_validacao = 800

# Repetições.
epochs = 50

# Número de exemplos de treinamento usados em uma repetição.
batch_size = 16

if K.image_data_format() == 'channels_first':
    formato_input = (3, img_largura, img_altura)
else:
    formato_input = (img_largura, img_altura, 3)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=formato_input)) # Aqui é criado uma Camada Convolucional 2D.
model.add(Activation('relu')) # Essa função de ativação chama-se Unidade Linear Retificada (ReLU) -- f(u)=max(0,u).
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid')) # Função de ativação Sigmoid -- f(x) = 1 / (1 + e ^ -x)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Augmentação utilizada pro treino. 
# Basicamente cria várias imagens aleatórias pra não haver repetições.
treino_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Augmentação utilizada pros testes. 
# Só escala a imagem.
teste_datagen = ImageDataGenerator(rescale=1. / 255)

# Preparando o treino da IA.
treino_generator = treino_datagen.flow_from_directory(
    pasta_treino,
    target_size=(img_largura, img_altura),
    batch_size=batch_size,
    class_mode='binary') # É binário pois só pode ser cachorro ou gato.

# Preparando a validação da IA.
validacao_generator = teste_datagen.flow_from_directory(
    pasta_validacao,
    target_size=(img_largura, img_altura),
    batch_size=batch_size,
    class_mode='binary') # É binário pois só pode ser cachorro ou gato.

# Cria o modelo, basicamente roda a IA.
model.fit_generator(
    treino_generator, # Treina a IA.
    steps_per_epoch=samples_de_treinamento // batch_size,
    epochs=epochs, # Número de repetições.
    validation_data=validacao_generator, # Valida a IA.
    validation_steps=samples_de_validacao // batch_size)

model.save_weights('first_try.h5')