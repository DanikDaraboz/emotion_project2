from flask import Flask, render_template_string, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === Загрузка модели и каскада ===
model = load_model('emotion_model5_v2_fastaccurate.h5')  # новая модель
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emotions = ['angry', 'happy', 'sad', 'surprise']  # порядок как в обучении
IMG_SIZE = 224  # новый размер

app = Flask(__name__)

# === Камера и предсказание ===
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_color = frame[y:y+h, x:x+w]
            if roi_color.size == 0:
                continue

            # Подготовка ROI для модели
            roi = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
            roi = roi / 255.0
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi)
            emotion = emotions[np.argmax(preds)]

            # Отрисовка
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string("""
        <html>
        <head><title>Emotion Detection</title></head>
        <body style="text-align:center; background:#111; color:white;">
            <h2>Real-time Emotion Detection 😊</h2>
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
        </body>
        </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
