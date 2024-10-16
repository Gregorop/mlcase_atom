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
                #сразу нормализуем оценку от 0 до 10
                score = int(filename[filename.find('_')+1])/10 
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

sentence = "Soo cute movie, nice actors"

# Преобразуем строку в тензор
input_tensor = tf.constant([sentence])

# Применяем слой векторизации
vectorized_output = vectorize_layer(input_tensor)

print(f"Original: {sentence}")
print(f"Vectorized: {vectorized_output.numpy()}")