"""
Car and pedestrian tracking using Open CV.
"""
import cv2


# Video
video = cv2.VideoCapture('video_example_car.mp4')

# Pre-trained car classifier / Clasificador de carro pre-entrenado
car_tracker = 'car_detector.xml'

# Car classifier / Clasificador de carros
car_tracker = cv2.CascadeClassifier(car_tracker)

# Loop until car stops
while True:
    # Read current frame / Leer el cuadro actual
    (read_successful, frame) = video.read()
    
    # Safe coding / Código seguro
    if read_successful:
        # Must convert to grayscale / Convertir a escala de grises
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect cars / Detectar carros
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    # Car coordinates / Coordenadas de carros
    print(cars)

    # Draw rectanlges around the cars / Dibujar rectángulos alrededor de los carros
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Display image with cars spotted / Mostrar imagen con los carros detectados
    cv2.imshow("\tCar Detector", frame)

    # Don't close. Wait here until a key is pressed / No cierra. Espera aquí hasta que una tecla es presionada  
    key = cv2.waitKey(1)
    
    # Stop if Q or q is pressed / Detener si Q o q es presionada -> 81 and 113 because of the ASCII characters
    if key == 81 or key == 113:
        break
    
# Release videocapture objetct / Liberar el objeto de video
video.release()

print("Code completed!")
