from flask import Flask, render_template, request, url_for, redirect
from os.path import join
import tensorflow as tf
from keras.models import load_model
import numpy as np
import uuid
import os
from datetime import date

app = Flask(__name__)
model = load_model('skinModel2.h5')
categories = ["Actinic Keratosiss", "Basal Cell Carcinoma", "Benign", "Dermatofibroma", "Melanoma", "Nevus", "Pigmented Benigh Keratosis", "Seborrheic Keratosis", "Squamous Cell Carcinoma", "Vascular Lesion"]

recommendations = [
    [
        "It is recommended to avoid sun exposure and use sunscreen daily to prevent further skin damage.",
        "Consult a dermatologist for possible treatments like cryotherapy, topical chemotherapy, or laser therapy.",
        "Regular follow-ups are essential to monitor any changes in the skin.",
        "Consider lifestyle changes to protect your skin, such as wearing protective clothing and avoiding tanning beds."
    ],
    [
        "Seek treatment from a dermatologist as soon as possible to evaluate the lesion.",
        "Options may include surgical excision, Mohs surgery, or topical chemotherapy, depending on the size and location.",
        "Regular skin checks are crucial to detect any new lesions early.",
        "Discuss reconstruction options with your dermatologist or plastic surgeon if surgical removal is needed."
    ],
    [
        "These lesions are typically harmless, but it's important to monitor them for any changes.",
        "Maintain a routine of regular skin checks with a dermatologist.",
        "If there are any changes in appearance, size, or color, consult a dermatologist."
    ],
    [
        "Usually benign and do not require treatment unless symptomatic; consult a dermatologist if they become bothersome.",
        "Consider excision if the dermatofibroma is causing discomfort or cosmetic concerns.",
        "Keep the area clean and monitor for any changes."
    ],
    [
        "Immediate medical attention is critical; contact a dermatologist or oncologist right away.",
        "Treatment may involve surgery, immunotherapy, or targeted therapy based on the stage of melanoma.",
        "Consider genetic testing to guide treatment decisions.",
        "Regular follow-ups are essential for monitoring."
    ],
    [
        "Monitor for any changes in size, shape, or color of the nevus.",
        "If there are changes, consult a dermatologist for evaluation and possible biopsy.",
        "Maintain regular skin checks to catch any suspicious changes early."
    ],
    [
        "These are usually harmless, but regular monitoring is advised.",
        "Consult a dermatologist if there are any concerns about changes in appearance.",
        "Consider treatment options if the keratosis becomes irritated or painful."
    ],
    [
        "Typically benign; if removal is desired for cosmetic reasons, consult a dermatologist.",
        "Regular monitoring for changes is important, especially if there are multiple lesions.",
        "Use sunscreen to protect your skin, as these growths can be more sensitive to sun exposure."
    ],
    [
        "Seek treatment promptly; options include surgical excision, radiation therapy, or topical chemotherapy.",
        "Discuss a comprehensive treatment plan with your dermatologist or oncologist.",
        "Regular follow-ups are necessary to monitor for recurrence."
    ],
    [
        "Consult a dermatologist for evaluation and treatment options, which may include laser therapy or sclerotherapy.",
        "Monitor for any changes in size or color, as some vascular lesions may require further investigation.",
        "Maintain a healthy lifestyle to support skin health."
    ]
]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def indexPost():
    userId = str(uuid.uuid4())
    UPLOADS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads', userId)
    os.makedirs(UPLOADS_PATH)
    file = request.files['imagefile']
    file_type = file.filename.split('.')[-1]
    file.save(os.path.join(UPLOADS_PATH, 'scan.'+file_type))
    return redirect('/'+userId)

@app.route('/<uuid>', methods=['GET'])
def showImage(uuid):
    UPLOADS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads', uuid)
    image_extensions = ['jpg', 'jpeg', 'png']
    image = None
    for ext in image_extensions:
        if os.path.exists(os.path.join(UPLOADS_PATH, 'scan.'+ext)):
            image = 'scan.'+ext
            break
    image_path = join(UPLOADS_PATH, image)
    image = url_for('static', filename='uploads/'+uuid+'/'+image)
    loaded_image =  tf.keras.utils.load_img(image_path, target_size=(224, 224))
    loaded_image = tf.keras.utils.img_to_array(loaded_image)
    loaded_image = loaded_image.reshape((1, loaded_image.shape[0], loaded_image.shape[1], loaded_image.shape[2]))
    predictions = model.predict(loaded_image)
    result = np.argmax(predictions, axis = 1)
    confidence = predictions[0][result]
    res = result[0]
    result = categories[result[0]]
    return render_template('response.html', image=image, result=result, confidence=confidence*100, recommendation = recommendations[res], date = date.today().strftime("%B %d, %Y"))

if __name__ == '__main__':
    app.run(port = 3000, debug=True)