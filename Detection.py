import sys
import cv2
import numpy as np


def preprocess_image(image):
    """Prétraite l'image pour améliorer la détection"""
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Amélioration du contraste avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return enhanced


def detect_barcode(detector, image):
    """Essaie de détecter un code-barres avec différentes techniques"""
    results = []
    
    # 1. Essai avec l'image originale
    result = detector.detectAndDecode(image)
    results.append(("Image originale", result))
    
    # 2. Essai avec l'image en niveaux de gris améliorée
    gray_enhanced = preprocess_image(image)
    result = detector.detectAndDecode(gray_enhanced)
    results.append(("Image améliorée", result))
    
    # 3. Essai avec l'image agrandie (si petite)
    height, width = image.shape[:2]
    if width < 800 or height < 600:
        scale = 2.0
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        result = detector.detectAndDecode(resized)
        results.append(("Image agrandie", result))
    
    # 4. Essai avec binarisation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result = detector.detectAndDecode(binary)
    results.append(("Image binarisée", result))
    
    # 5. Essai avec égalisation d'histogramme
    equalized = cv2.equalizeHist(gray)
    result = detector.detectAndDecode(equalized)
    results.append(("Histogramme égalisé", result))
    
    return results


def parse_detection_result(result):
    """Parse le résultat de detectAndDecode quelle que soit sa forme"""
    retval = False
    decoded_info = []
    decoded_type = []
    corners = None

    if isinstance(result, tuple):
        if len(result) == 4:
            # Signature : retval, decoded_info, decoded_type, corners
            retval, decoded_info, decoded_type, corners = result
        elif len(result) == 3:
            # Signature : decoded_info, decoded_type, corners
            decoded_info, decoded_type, corners = result
            if decoded_info is not None and len(decoded_info) > 0:
                retval = any(info for info in decoded_info)
    
    return retval, decoded_info, decoded_type, corners


def main():
    if len(sys.argv) < 2:
        print("Usage : python3 Detection.py chemin/vers/image.png")
        sys.exit(1)

    image_path = sys.argv[1]

    # Chargement de l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Impossible de charger l'image : {image_path}")
        sys.exit(1)

    print(f"Image chargée : {image.shape[1]}x{image.shape[0]} pixels")

    # Création du détecteur de codes-barres
    try:
        detector = cv2.barcode_BarcodeDetector()
    except AttributeError:
        print(
            "Erreur : votre version d'OpenCV ne contient pas "
            "cv2.barcode_BarcodeDetector.\n"
            "Installez : pip install opencv-contrib-python"
        )
        sys.exit(1)

    print("\nTentatives de détection avec différentes méthodes...\n")

    # Essayer plusieurs méthodes de détection
    all_results = detect_barcode(detector, image)
    
    # Chercher la première détection réussie
    best_result = None
    best_method = None
    
    for method_name, result in all_results:
        retval, decoded_info, decoded_type, corners = parse_detection_result(result)
        
        if retval and decoded_info and len(decoded_info) > 0 and any(info for info in decoded_info):
            best_result = (retval, decoded_info, decoded_type, corners)
            best_method = method_name
            print(f"✓ Détection réussie avec : {method_name}")
            break
        else:
            print(f"✗ Échec avec : {method_name}")

    # Traitement du résultat
    if best_result is None:
        print("\n❌ Aucun code-barres détecté avec aucune méthode.")
        print("\nConseils :")
        print("  - Vérifiez que l'image contient bien un code-barres visible")
        print("  - Assurez-vous que le code-barres est net et bien éclairé")
        print("  - Le code-barres ne doit pas être trop petit ou déformé")
        print("  - OpenCV supporte : EAN-8, EAN-13, UPC-A, UPC-E")
        print("  - Vérifiez votre version : pip install --upgrade opencv-contrib-python")
    else:
        retval, decoded_info, decoded_type, corners = best_result
        
        print(f"\n✓ Code(s)-barres détecté(s) avec : {best_method}")
        print()
        
        display_image = image.copy()
        
        for i, info in enumerate(decoded_info):
            if not info:
                continue

            print(f"  Code-barres #{i+1}:")
            print(f"    Valeur : {info}")
            if decoded_type is not None and i < len(decoded_type):
                print(f"    Type   : {decoded_type[i]}")

            # Récupération des coins du code-barres
            if corners is not None and i < len(corners):
                pts = corners[i]
                
                # Debug : afficher la forme originale
                print(f"    Corners shape : {pts.shape}")
                
                # Assurer que pts est un tableau 2D de points
                if len(pts.shape) == 3:
                    pts = np.squeeze(pts)  # (1, n, 2) -> (n, 2)
                
                # Vérifier qu'on a bien des points valides
                if pts.shape[0] < 3:
                    print(f"    Pas assez de points pour dessiner")
                    continue
                
                # Convertir en entiers pour le dessin
                pts_int = pts.astype(np.int32)
                
                # Calcul du rectangle englobant
                x, y, w, h = cv2.boundingRect(pts_int)
                
                print(f"    Rectangle : x={x}, y={y}, w={w}, h={h}")
                
                # Vérifier que le rectangle est valide
                if w <= 0 or h <= 0:
                    print(f"    Rectangle invalide")
                    continue
                
                # Dessiner le rectangle vert
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                # Dessiner aussi le contour exact en bleu
                cv2.polylines(display_image, [pts_int], True, (255, 0, 0), 2)

                # Dessiner les chiffres du code-barres individuellement
                # Calculer l'espacement entre chaque chiffre
                barcode_digits = str(info)
                num_digits = len(barcode_digits)
                
                if num_digits > 0:
                    # Calculer la largeur disponible pour chaque chiffre
                    digit_width = w / num_digits
                    
                    # Calculer une taille de police adaptée
                    font_scale = min(w / (num_digits * 30), h / 50, 1.2)
                    font_thickness = max(1, int(font_scale * 2))
                    
                    # Position Y pour les chiffres (sous le code-barres)
                    digit_y = y + h + int(25 * font_scale)
                    
                    # Dessiner chaque chiffre
                    for idx, digit in enumerate(barcode_digits):
                        # Position X pour ce chiffre
                        digit_x = x + int(idx * digit_width + digit_width / 4)
                        
                        # Obtenir la taille du chiffre
                        (dw, dh), _ = cv2.getTextSize(
                            digit, 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            font_scale, 
                            font_thickness
                        )
                        
                        # Fond blanc pour le chiffre
                        cv2.rectangle(
                            display_image,
                            (digit_x - 2, digit_y - dh - 2),
                            (digit_x + dw + 2, digit_y + 4),
                            (255, 255, 255),
                            -1
                        )
                        
                        # Dessiner le chiffre en noir
                        cv2.putText(
                            display_image,
                            digit,
                            (digit_x, digit_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (0, 0, 0),
                            font_thickness,
                            cv2.LINE_AA,
                        )
                
                # Texte du code complet au-dessus
                text = f"{info}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                
                text_y = y - 10 if y - 10 > text_height else y + h + 50
                
                cv2.rectangle(
                    display_image,
                    (x, text_y - text_height - 5),
                    (x + text_width + 5, text_y + baseline),
                    (0, 255, 0),
                    -1
                )
                
                cv2.putText(
                    display_image,
                    text,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        # Affichage de l'image
        cv2.imshow("Detection code-barres", display_image)
        print("\n💡 Appuyez sur une touche dans la fenêtre d'image pour quitter.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()