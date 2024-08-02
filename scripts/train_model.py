# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# dataset_path = 'data/plantvillage'
# datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# train_gen = datagen.flow_from_directory(
#     dataset_path,
#     target_size=(256, 256),
#     batch_size=32,
#     class_mode='categorical',
#     subset='training'
# )
# val_gen = datagen.flow_from_directory(
#     dataset_path,
#     target_size=(256, 256),
#     batch_size=32,
#     class_mode='categorical',
#     subset='validation'
# )
# from tensorflow.keras.models import Sequential 
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(train_gen.num_classes, activation='softmax')
# ])
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(train_gen, epochs=25, validation_data=val_gen)
# val_loss, val_acc = model.evaluate(val_gen)
# print(f'Validation accuracy: {val_acc:.2f}')
# model.save('models/plant_disease_model.h5')
#
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


dataset_path = 'data/plantvillage'
model_save_path = 'models/plant_disease_model.h5'

# Checando se os diretórios contem imagens
classes = ['Strawberry___healthy', 'Strawberry___Leaf_scorch']
for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    if not os.path.exists(class_path):
        raise ValueError(f"Directory {class_path} does not exist.")
    else:
        files = os.listdir(class_path)
        if not files:
            raise ValueError(f"No images found in directory {class_path}.")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 
)

# Create train and validation generators
train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    classes=classes
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    classes=classes
)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

# compilando o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# treinando o modelo
history = model.fit(train_gen, epochs=25, validation_data=val_gen)


val_loss, val_acc = model.evaluate(val_gen)
print(f'Validation accuracy: {val_acc:.2f}')

# salvando
model.save(model_save_path)
