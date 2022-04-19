import numpy as np
from flask import Flask, render_template, request
from keras.preprocessing import image
#from keras.preprocessing.image import load_img
from keras.models import load_model
from keras.models import model_from_json
import PIL 
from PIL import Image


# Create flask app
app = Flask(__name__)
# model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=['GET'])
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    #model = load_model("model_Classifier.h5")
    json_file = open('model.json','r')
    loaded_model_json=json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #load weights into new model
    loaded_model.load_weights("model_weights.h5")
    print("Loaded Model from disk")
    #compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    img=image.load_img(image_path ,target_size=(224,224))
    test_image=image.img_to_array(img)
    test_image=np.expand_dims(test_image, axis = 0)
    dictionary = {0:'100',1:'200',2:'2000',3:'500',4:'50',5:'10',6:'20'}
    result = loaded_model.predict(test_image)
    max_value = np.argmax(result)
    result = dictionary[max_value]

    return render_template("index.html", prediction_text = result)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')