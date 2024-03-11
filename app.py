from flask import Flask, render_template, redirect, url_for, Response, flash, request
import cv2
import time
import os
import shutil
import numpy as np
from keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'captured_images'
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


model = load_model('model1.h5')

camera = cv2.VideoCapture(0)


def preprocess_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (48, 48))
    img_normalized = img_resized.astype('float32') / 255.0
    return np.expand_dims(img_normalized, axis=-1)

def predict_emotions(images_folder):
    predictions = []
    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(images_folder, filename))
            img = preprocess_image(img)
            prediction = model.predict(np.array([img]))
            predictions.append(prediction[0])
    return np.mean(predictions, axis=0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    duration = 10  
    captured_images_folder = capture_frames(duration)
    camera.release()
    cv2.destroyAllWindows()
    flash('Capture and prediction completed successfully!')
    return redirect(url_for('result', folder=captured_images_folder))

def capture_frames(duration):
    images_folder = app.config['UPLOAD_FOLDER']
    
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    start_time = time.time()
    img_count = 0

    while (time.time() - start_time) < duration:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(images_folder, f'face_{img_count}.jpg'), face)
            img_count += 1

        time.sleep(0.5)

    return images_folder

@app.route('/result')
def result():
    folder = request.args.get('folder')
    if folder:
        predictions = predict_emotions(folder)
        shutil.rmtree(folder)
        return render_template('result.html', predictions=predictions)
    else:
        return "No captured images folder provided."

if __name__ == "__main__":
    app.run(debug=True)
