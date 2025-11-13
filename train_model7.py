# train_model_v2_sota.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import cv2

# ==========================
# 🔧 НАСТРОЙКА ОКРУЖЕНИЯ
# ==========================
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

print("\n=== TensorFlow Info ===")
print("Версия TensorFlow:", tf.__version__)
print("CUDA доступен:", tf.test.is_built_with_cuda())
print("Доступные GPU:", tf.config.list_physical_devices('GPU'))

# ==========================
# 📁 ПУТИ К ДАННЫМ
# ==========================
train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'

# ==========================
# ⚙️ НАСТРОЙКИ
# ==========================
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 15

# ==========================
# 📈 SOTA-АУГМЕНТАЦИИ (ArtAug-подобные)
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    shear_range=0.15,
    channel_shift_range=20.0,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_data = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
test_data = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ==========================
# 🧠 СОЗДАНИЕ МОДЕЛИ
# ==========================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # сначала заморозим

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='swish')(x)
x = Dropout(0.3)(x)
output = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# ==========================
# 📦 КОМПИЛЯЦИЯ (Этап 1)
# ==========================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks_stage1 = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6),
    ModelCheckpoint("best_model_sota_stage1.h5", save_best_only=True)
]

print("\n=== ЭТАП 1: Базовое обучение (замороженные слои) ===")
model.fit(train_data, validation_data=val_data, epochs=EPOCHS_STAGE1, callbacks=callbacks_stage1, verbose=1)

# ==========================
# 🔧 FINE-TUNING (Этап 2)
# ==========================
print("\n=== ЭТАП 2: Тонкая настройка последних слоёв ===")
base_model.trainable = True
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

callbacks_stage2 = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7),
    ModelCheckpoint("best_model_sota_finetuned.h5", save_best_only=True)
]

model.fit(train_data, validation_data=val_data, epochs=EPOCHS_STAGE2, callbacks=callbacks_stage2, verbose=1)

# ==========================
# 🧩 ОЦЕНКА
# ==========================
print("\nОЦЕНКА ТОЧНОСТИ:")
loss, acc = model.evaluate(test_data)
print(f"✅ Финальная точность: {acc*100:.2f}%")

# ==========================
# 💾 СОХРАНЕНИЕ
# ==========================
model.save("emotion_model_sota.h5")
print("\n🚀 Модель сохранена: emotion_model_sota.h5")

# ==========================
# 🔥 Grad-CAM для explainability
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
