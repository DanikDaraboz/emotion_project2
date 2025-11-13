from flask import Flask, render_template_string, Response
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==========================
# 🔥 Загрузка модели и каскада
# ==========================
model = load_model('emotion_model_sota.h5')  # новая SOTA модель
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emotions = ['angry', 'happy', 'sad', 'surprise']
IMG_SIZE = 224

app = Flask(__name__)

# ==========================
# 🔥 Grad-CAM функция
# ==========================
def gradcam(model, img_array, layer_name='Conv_1'):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (tf.reduce_max(heatmap)+1e-8)
    heatmap = cv2.resize(heatmap.numpy(), (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

# ==========================
# 🔥 Камера и предсказание
# ==========================
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
            roi_input = roi / 255.0
            roi_input = np.expand_dims(roi_input, axis=0)

            preds = model.predict(roi_input)
            emotion = emotions[np.argmax(preds)]

            # Grad-CAM overlay
            heatmap = gradcam(model, roi_input)
            overlay = cv2.addWeighted(roi, 0.6, heatmap, 0.4, 0)

            # Вставка overlay обратно в кадр
            overlay_resized = cv2.resize(overlay, (w, h))
            frame[y:y+h, x:x+w] = overlay_resized

            # Подпись эмоции
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==========================
# 🔥 Web Routes
# ==========================
@app.route('/')
def index():
    return render_template_string("""
        <html>
        <head><title>Emotion Detection SOTA</title></head>
        <body style="text-align:center; background:#111; color:white;">
            <h2>Real-time Emotion Detection with Grad-CAM 🔥</h2>
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
            <p style="font-size:14px; color:gray;">
                All predictions are AI-generated for research purposes only. Faces are processed locally.
            </p>
        </body>
        </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================
# 🔥 Запуск
# ==========================
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
