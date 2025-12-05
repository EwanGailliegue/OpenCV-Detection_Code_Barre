import cv2
import numpy as np


def detect_barcodes(frame):
    """
    Détecte des régions ressemblant à des codes-barres dans une image couleur.
    Retourne la liste des bounding boxes : [(x, y, w, h), ...]
    """
    # 1) Niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2) Gradient horizontal - vertical (les barres sont verticales)
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # 3) Lissage + seuillage
    blurred = cv2.blur(gradient, (9, 9))
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # 4) Fermeture pour regrouper les lignes du code-barres
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 5) Érosions / dilatations pour nettoyer
    closed = cv2.erode(closed, None, iterations=3)
    closed = cv2.dilate(closed, None, iterations=3)

    # 6) Recherche des contours
    contours, _ = cv2.findContours(
        closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for c in contours:
        # Filtre sur la taille
        area = cv2.contourArea(c)
        if area < 1000:  # à ajuster selon vos images
            continue

        x, y, w, h = cv2.boundingRect(c)

        # Filtre sur le ratio largeur/hauteur (code-barres = oblong)
        aspect_ratio = w / float(h)
        if aspect_ratio < 2.0:  # un peu arbitraire, à ajuster
            continue

        boxes.append((x, y, w, h))

    return boxes


def main():
    image = cv2.imread("img/2.png")
    if image is None:
        print("Impossible de charger l'image.")
        return

    boxes = detect_barcodes(image)

    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Detection de codes-barres", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
