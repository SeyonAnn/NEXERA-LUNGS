import numpy as np

CLASS_NAMES = ['Normal', 'Pneumonia']

def predict(model, img_array):
    prediction = model.predict(img_array, verbose=0)
    confidence = float(prediction[0][0])
    if confidence >= 0.5:
        label = 'PNEUMONIA'
        score = confidence * 100
    else:
        label = 'NORMAL'
        score = (1 - confidence) * 100
    normal_prob = (1 - confidence) * 100
    pneumonia_prob = confidence * 100
    return label, score, normal_prob, pneumonia_prob