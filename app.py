from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from pickle import load
from numpy import argmax

# Keras
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet import preprocess_input
from keras.models import Model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/pet_name_model.h5'
# load the model
model = load_model('models/pet_name_model.h5')
model._make_predict_function()          # Necessary

# extract features from each photo in the directory
def extract_features(filename):
    # load the model
    imfeature_model = MobileNet(weights='models/mobilenet_1_0_224_tf.h5')
    # re-structure the model
    imfeature_model.layers.pop()
    imfeature_model = Model(inputs=imfeature_model.inputs, outputs=imfeature_model.layers[-1].output)
    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = imfeature_model.predict(image, verbose=0)
    return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text

#Runs all the sub functions and cleans and returns the text of the name
def return_clean_desc(photo_path,model,max_length=8):
    tokenizer = load(open('models/tokenizer.pkl', 'rb'))
    
    # load and prepare the photograph
    photo = np.squeeze(np.squeeze(extract_features(photo_path),axis=0),axis=0)

    # generate description
    description = generate_desc(model, tokenizer, photo, max_length)

    #Clean up description
    cleaned_description = re.search(r'seq (.*?) end',description).group(1).title()
    
    return cleaned_description


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        pet_name = return_clean_desc(file_path,model)
        os.remove(file_path)

        return pet_name
    return None


if __name__ == '__main__':

    #serve the app
    port = int(os.environ.get("PORT",33507))
    app.run(host='0.0.0.0',port=port,debug=False,threaded=False)

