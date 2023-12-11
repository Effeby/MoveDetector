import cv2
import pygame

# Initialisez Pygame pour le son
pygame.mixer.init()

# Chargez le son d'alerte
alert_sound = pygame.mixer.Sound('alert.wav')  # Assurez-vous d'avoir un fichier audio nommé 'alert.wav' dans le même répertoire

# Initialisez la webcam
cap = cv2.VideoCapture(0)

# Chargez le modèle HOG pour la détection des humains
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Variables pour la détection de mouvement
threshold = 500  # Ajustez ce seuil selon vos besoins
prev_frame = None

while True:
    # Récupérez le cadre actuel
    ret, frame = cap.read()

    # Convertissez le cadre en niveaux de gris pour la détection HOG
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détectez des humains dans l'image
    humans, _ = hog.detectMultiScale(gray)

    # Dessinez des rectangles autour des humains détectés
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Si c'est la première frame ou la différence est significative, considérez comme un mouvement
    if prev_frame is not None:
        diff = cv2.absdiff(prev_frame, gray)
        _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > threshold:
                # Encadrez en rouge les mouvements détectés
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 2)

                # Émettre un son d'alerte
                alert_sound.play()
                break

    # Affichez la vidéo en direct avec les rectangles autour des humains et le suivi
    cv2.imshow("Detection de mouvement sur humains", frame)

    # Mettez à jour la frame précédente
    prev_frame = gray

    # Quittez la boucle en appuyant sur la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérez les ressources
cap.release()

cv2.destroyAllWindows()
