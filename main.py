# Importar as bibliotecas necessárias
import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Inicializar o detector de mãos do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Função para detectar mãos em uma imagem
def detect_hands(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    return image

# Carregar imagens de referência
known_hands = []
known_labels = []
labels = ['A', 'B', 'C']  # Lista de rótulos para as letras A, B e C

for label in labels:
    dir_path = os.path.join("imagens", label)
    if os.path.isdir(dir_path):
        for image in os.listdir(dir_path):
            image_path = os.path.join(dir_path, image)
            hand_image = cv2.imread(image_path)
            hand_image = detect_hands(hand_image)
            known_hands.append(hand_image)
            known_labels.append(label)

# Pré-processar imagens de referência
known_hands_encoded = []
for hand in known_hands:
    hand = cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)
    hand = cv2.resize(hand, (160, 160))
    known_hands_encoded.append(hand)

known_hands_encoded = np.array(known_hands_encoded) / 255.0

# Codificar rótulos
label_encoder = tf.keras.utils.to_categorical(np.array([labels.index(label) for label in known_labels]))

# Criar modelo de reconhecimento de mãos
inputs = tf.keras.Input(shape=(160, 160, 3))
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(len(labels), activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(known_hands_encoded, label_encoder, epochs=30, batch_size=32)

# Inicializar a câmera
video = cv2.VideoCapture(0)

while True:
    _, img = video.read()
    img = detect_hands(img)

    cv2.imshow("Resultado", img)
    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
