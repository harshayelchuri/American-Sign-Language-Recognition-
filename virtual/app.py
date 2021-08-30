from flask import Flask,render_template,request

import speech_recognition as sr



app = Flask(__name__, static_folder='./static')
@app.route('/',methods=["GET","POST"])
def Hello():
    if request.method=="POST":
        fucn()
        return render_template("index.html")
    else:
        return render_template("index.html")

@app.route('/image',methods=["POST"])
def text_to_image():
    if request.method=="POST":
        text=request.form.get("text")
        print(text)
        # text_to_speech(text)
        return render_template("image.html",text=text.upper(),len=len(text))

@app.route('/speech',methods=["POST"])
def speech_to_image():
    print('hi')
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak Anything :")
        r.adjust_for_ambient_noise(source,duration=1)
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print(text)
            return render_template("image.html",text=text.upper(),len=len(text))
        except:
            print("Sorry could not recognize what you said")
    return render_template("image.html",text=text.upper(),len=len(text))

def fucn():
    import cv2
    import numpy as np

    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    data_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

    MODEL_NAME = 'asl.h5'
    model = load_model(MODEL_NAME)

    IMAGE_SIZE = 200
    CROP_SIZE = 400

    classes_file = open("classes.txt")
    classes_string = classes_file.readline()
    classes = classes_string.split()
    classes.sort()

    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()

        cv2.rectangle(frame, (0, 0), (CROP_SIZE, CROP_SIZE), (0, 255, 0), 3)

        cropped_image = frame[0:CROP_SIZE, 0:CROP_SIZE]
        resized_frame = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))
        reshaped_frame = (np.array(resized_frame)).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        frame_for_model = data_generator.standardize(np.float64(reshaped_frame))

        prediction = np.array(model.predict(frame_for_model))
        predicted_class = classes[prediction.argmax()]      # Selecting the max confidence index.
        print(predicted_class)

        prediction_probability = prediction[0, prediction.argmax()]
        if prediction_probability > 0.5:
            cv2.putText(frame, '{} - {:.2f}%'.format(predicted_class, prediction_probability * 100),(10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)
            print(predicted_class)

        elif prediction_probability > 0.2 and prediction_probability <= 0.5:
            cv2.putText(frame, 'Maybe {}... - {:.2f}%'.format(predicted_class, prediction_probability * 100), (10, 450), 1, 2, (0, 255, 255), 2, cv2.LINE_AA)
            print(predicted_class)

        else:
            cv2.putText(frame, classes[-2], (10, 450), 1, 2, (255, 255, 0), 2, cv2.LINE_AA)
            print(predicted_class)
        cv2.imshow('frame', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    app.run(port=4996,debug=True, use_reloader=True)
