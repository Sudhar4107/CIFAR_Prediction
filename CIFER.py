from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import joblib
import tensorflow
import numpy as np

app = Flask(__name__)

@app.route('/')
def Home ():
    return render_template('form.html')

count=0
@app.route('/submit', methods=['POST'])
def Predict():
    global count
    img = request.files['image']
    pic_loc=rf'C:\Users\ELCOT\Desktop\VsCode_Practice\deployment_env\CIFER\pred_img\pred_img{count}.jpg'
    model_loc=r'C:\Users\ELCOT\Desktop\VsCode_Practice\deployment_env\CIFER\CIFER.h5'

    img.save(pic_loc)
    
    classes=['aeroplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    test_img = image.load_img(pic_loc, target_size=(32,32))
    test_img = image.img_to_array(test_img)
    test_img = test_img/255
    test_img= test_img.reshape(1,32,32,3)
  
    model = tensorflow.keras.models.load_model(model_loc)

    result = model.predict(test_img)
    pred= classes[np.argmax(result[0])]

    return render_template('predict.html',data=f'It is a {pred}')

if(__name__=='__main__'):
    app.run(host='0.0.0.0', port=5000, debug= True)