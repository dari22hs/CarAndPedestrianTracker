"""
Car and pedestrian tracking using Open CV.
"""
import cv2


# Image / Imagen
img_file = 'car_example.jpg'

# Pre-trained car classifier / Clasificador de carro pre-entrenado
classifier_file = 'car_detector.xml'

# Create OpenCV image / Crear imagen OpenCV
img = cv2.imread(img_file)

# Convert to grayscale / Convertir a escala de grises
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Car classifier / Clasificador de carros
car_tracker = cv2.CascadeClassifier(classifier_file)

# Detect cars / Detectar carros
cars = car_tracker.detectMultiScale(grayscale_img)
# Car coordinates / Coordenadas de carros
print(cars)

# Draw rectanlges around the cars / Dibujar rectángulos alrededor de los carros
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Display image with cars spotted / Mostrar imagen con los carros detectados
cv2.imshow("\tCar Detector", img)

# Don't close. Wait here until a key is pressed / No cierra. Espera aquí hasta que una tecla es presionada
cv2.waitKey()

print("Code completed!")
