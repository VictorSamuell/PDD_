# from flask import Flask, request, render_template
# from tensorflow.keras.models import load_model
# import numpy as np
# from tensorflow.keras.preprocessing import image
# import os

# app = Flask(__name__)
# model = load_model('models/plant_disease_model.h5')

# def predict_image(img_path):
#     img = image.load_img(img_path, target_size=(256, 256))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0
#     prediction = model.predict(img_array)
#     class_idx = np.argmax(prediction, axis=1)
#     class_labels = {0: 'Strawberry__Healthy', 1: 'Strawberry__Leaf_scorch'}
#     return class_labels[class_idx[0]]

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         img_file = request.files['file']
#         if not os.path.exists('static/uploads'):
#             os.makedirs('static/uploads')
#         img_path = os.path.join('static/uploads', img_file.filename)
#         img_file.save(img_path)
#         predicted_class = predict_image(img_path)
#         return f'Predicted Class: {predicted_class}'
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
model = load_model('models/plant_disease_model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)
    class_labels = {0: 'Folha Saudavel', 1: 'Folha Doente'}
    return class_labels[class_idx[0]]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    if request.method == 'POST':
        img_file = request.files['file']
        if not os.path.exists('static/uploads'):
            os.makedirs('static/uploads')
        filename = img_file.filename
        img_path = os.path.join('static/uploads', filename)
        img_file.save(img_path)
        prediction = predict_image(img_path)
    return render_template('index.html', prediction=prediction, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)