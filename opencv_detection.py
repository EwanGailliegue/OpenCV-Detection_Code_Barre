import cv2
import numpy as np

print(cv2.__version__)  # au cazou

# Chargement de l'image
img = cv2.imread('img/image.png')
if img is None:
    raise FileNotFoundError("Impossible de charger l’image 'img/manga.jpg'")

# Création du détecteur de codes-barres
bd = cv2.barcode.BarcodeDetector()

# On utilise detectAndDecodeWithType pour avoir: retval, info, type, points
if hasattr(bd, "detectAndDecodeWithType"):
    retval, decoded_info, decoded_type, points = bd.detectAndDecodeWithType(img)
else:
    
    retval, decoded_info, decoded_type, points = bd.detectAndDecode(img)

print("retval        :", retval)
print("decoded_info  :", decoded_info)   # liste de chaînes
print("decoded_type  :", decoded_type)   # liste de types ('EAN-13', etc.)
print("type(points)  :", type(points))
print("points shape  :", None if points is None else points.shape)

# Vérification de la détection
if (not retval) or points is None or len(points) == 0:
    print("Aucun code-barres détecté.")
else:
    # points est en général de forme (N, 4, 2)
    points = np.array(points, dtype=np.int32)

    # Si un seul code avec forme (4, 2), on ajoute une dimension
    if points.ndim == 2:
        points = points.reshape(1, 4, 2)

    # Dessiner les polygones autour des codes-barres
    img = cv2.polylines(img, points, True, (0, 255, 0), 10)

    # Écrire le texte (valeur du code-barres) près de chaque code
    for text, pts, typ in zip(decoded_info, points, decoded_type):
        # ordre: [bottomLeft, topLeft, topRight, bottomRight]
        x, y = pts[1]  # topLeft
        label = f"{text} ({typ})"
        img = cv2.putText(
            img,
            label,
            (int(x) , int(y) + 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    ok = cv2.imwrite('output_img/barcode_opencv.jpg', img)
    print("Image sauvegardée :", ok)
