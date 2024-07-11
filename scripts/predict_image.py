# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# model = load_model('models/plant_disease_model.h5')
# # Dicionário para mapeamento das classes
# classes = {0: 'Healthy', 1: 'Leaf Scorch'}
# def predict_image(img_path):
#     img = image.load_img(img_path, target_size=(256, 256))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0
#     prediction = model.predict(img_array)
#     class_idx = int(prediction[0] > 0.5)
#     return classes[class_idx]
# if __name__ == "__main__":
#     img_path = 'path/to/test_image.jpg'
#     predicted_class = predict_image(img_path)
#     print(f'Predicted Class: {predicted_class}')

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Carregando o modelo
model = load_model('models/plant_disease_model.h5')


def predict_image(img_path):
    
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # previsao
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)
    
    # indices para classes
    class_labels = {0: 'Strawberry__Healthy', 1: 'Strawberry__Leaf_scorch'}
    return class_labels[class_idx[0]]


if __name__ == "__main__":

    # imagem para teste
    test_image_path = 'data\plantvillage\Strawberry___Leaf_scorch\0a08af15-adfe-447c-8ed4-17ed2702d810___RS_L.Scorch 0054.JPG'  
    
    
    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"A imagem de teste não foi encontrada: {test_image_path}")
    
    # fazendo a previsao com a imagem de teste
    predicted_class = predict_image(test_image_path)
    print(f'Predicted Class: {predicted_class}')
    
