from flask import Flask, render_template, request
from keras.models import load_model
from audio_utils import process_audio_to_spectrogram
from image_utils import preprocess_image
import os
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER_AUDIO'] = 'uploads/audio'
app.config['UPLOAD_FOLDER_IMAGE'] = 'uploads/image'

audio_model = load_model('models/audio_model.h5')
image_model = load_model('models/image_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    file = request.files['audio_file']
    path = os.path.join(app.config['UPLOAD_FOLDER_AUDIO'], file.filename)
    file.save(path)

    audio_input = process_audio_to_spectrogram(path)
    prediction = audio_model.predict(audio_input)
    result = f"Audio Prediction: Class {np.argmax(prediction)}"

    return render_template("index.html", audio_result=result)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    file = request.files['image_file']
    path = os.path.join(app.config['UPLOAD_FOLDER_IMAGE'], file.filename)
    file.save(path)

    image_input = preprocess_image(path)
    prediction = image_model.predict(image_input)
    result = f"Image Prediction: Class {np.argmax(prediction)}"

    return render_template("index.html", image_result=result)

if __name__ == '__main__':
    app.run(debug=True)
