import os, re, string
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import keras
import tensorflow as tf
import numpy as np

def clear_text(input_data):
    lowercase = tf.strings.lower(input_data)
    #конкретно "<br />"
    no_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    #убираем все знаки препинания, регуляркой типа [.,:!], каждый символ подойдет
    no_punct = tf.strings.regex_replace(no_html, f"[{re.escape(string.punctuation)}]", "")
    return no_punct

def load_data(data_dir):
    texts = []
    labels = []
    
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                #вернул цифорку, чтобы сделать многоклассовую
                score = int(filename[filename.find('_')+1])
                file_path = os.path.join(class_dir, filename)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                texts.append(text)
                labels.append(score)

    return texts, labels

data_dir = 'aclImdb_v1/train'
texts, labels = load_data(data_dir)

# Преобразование в тензоры для tf, попарно склеиваем текст и оценку
dataset = tf.data.Dataset.from_tensor_slices((texts, labels))

# Разделение на обучение и валидацию
train_size = int(0.8 * len(dataset))
raw_train = dataset.take(train_size)
raw_val = dataset.skip(train_size)

batch_size = 32  # делаем батчи как в text_dataset_from_directory
raw_train = raw_train.batch(batch_size)
raw_val = raw_val.batch(batch_size)


#параметры из гайда
max_features = 20000 #количество слов учитываемых при обучении
embedding_dim = 128 #каждое слово заменится на вектор из 128 интов
sequence_length = 500 #длина к которой приводятся комментарии

vectorize_layer = keras.layers.TextVectorization(
    standardize=clear_text,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length)


text_ds = raw_train.map(lambda x, y: x) #взяли только строчки из датасета
vectorize_layer.adapt(text_ds) #настраиваем словарь


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

#применяем vectorize_layer на датасет
train_ds = raw_train.map(vectorize_text)
val_ds = raw_val.map(vectorize_text)
#test_ds = raw_test.map(vectorize_text)



#из гайда
# Do async prefetching / buffering of the data for best performance on GPU.
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
#test_ds = test_ds.cache().prefetch(buffer_size=10)


from keras import layers
inputs = keras.Input(shape=(None,), dtype="int64")

# количество слов на размер вектора для 1 слова
x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)

x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

# всякие слои, пока оставил так
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

predictions = layers.Dense(10, activation="softmax", name="predictions")(x)

model = keras.Model(inputs, predictions)


#из гайда binary_crossentropy поменял на категории
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])

epochs = 3

model.fit(train_ds, validation_data=val_ds, epochs=epochs)
