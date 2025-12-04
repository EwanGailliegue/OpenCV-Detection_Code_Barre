import cv2
import numpy as np

# ----------------------------
# Mini-OCR maison / placeholder
# ----------------------------
def extract_digits(roi):
    """
    Extraction rudimentaire de chiffres 0-9 d'une image ROI.
    Pour l'instant: placeholder, retourne "" si aucun chiffre détecté.
    Vous pouvez remplacer par votre mini-OCR maison.
    """
    # Ici on pourrait appeler extract_digits maison ou template matching
    return ""  # Placeholder : aucun chiffre détecté

# ----------------------------
# Fonctions utilitaires
# ----------------------------
def rect_iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    if union == 0:
        return 0
    return inter / union

def merge_rects(rects, iou_thresh=0.12):
    if not rects:
        return []

    rects = rects.copy()
    merged = []
    used = [False]*len(rects)

    for i in range(len(rects)):
        if used[i]:
            continue
        x,y,w,h = rects[i]
        ux1, uy1, ux2, uy2 = x, y, x+w, y+h
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(rects)):
                if used[j]: 
                    continue
                rx, ry, rw, rh = rects[j]
                if rect_iou((ux1, uy1, ux2-ux1, uy2-uy1), (rx, ry, rw, rh)) > iou_thresh:
                    ux1 = min(ux1, rx)
                    uy1 = min(uy1, ry)
                    ux2 = max(ux2, rx+rw)
                    uy2 = max(uy2, ry+rh)
                    used[j] = True
                    changed = True
        merged.append((ux1, uy1, ux2-ux1, uy2-uy1))
    return merged

# ----------------------------
# Pipeline principal
# ----------------------------
img = cv2.imread("codebarre1.jpg")
if img is None:
    raise SystemExit("Impossible de lire sample.jpg — vérifiez le chemin.")

orig = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_img, w_img = gray.shape

# Amélioration contraste local
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray_eq = clahe.apply(gray)

# Gradient orienté horizontal
gradX = cv2.Sobel(gray_eq, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
gradY = cv2.Sobel(gray_eq, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
gradient = cv2.subtract(np.absolute(gradX), np.absolute(gradY))
gradient = cv2.convertScaleAbs(gradient)

# Blur + seuil Otsu
blurred = cv2.GaussianBlur(gradient, (9,9), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphologie : fermeture
kernel_w = max(21, img.shape[1]//40)
kernel_h = max(7, img.shape[0]//200)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Ouverture pour retirer bruit
closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)

# Dilatation pour joindre fragments
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9,kernel_w//2), 3))
closed = cv2.dilate(closed, dilate_kernel, iterations=2)

# Erosion douce
closed = cv2.erode(closed, None, iterations=1)

# Contours
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
candidates = []

for c in contours:
    area = cv2.contourArea(c)
    if area < 500:
        continue
    x,y,w,h = cv2.boundingRect(c)
    aspect = float(w)/float(h) if h>0 else 0
    rect_area = w*h
    solidity = area / rect_area if rect_area>0 else 0
    if aspect < 1.2 or solidity < 0.3 or h<10 or w<30:
        continue
    candidates.append((x,y,w,h))

# Fusionner rectangles proches
merged = merge_rects(candidates, iou_thresh=0.12)

# Expansion légère des boîtes
expanded = []
pad_x = int(max(2, w_img*0.01))
pad_y = int(max(2, h_img*0.005))
for (x,y,w,h) in merged:
    nx = max(0, x - pad_x)
    ny = max(0, y - pad_y)
    nw = min(w_img - nx, w + 2*pad_x)
    nh = min(h_img - ny, h + 2*pad_y)
    expanded.append((nx, ny, nw, nh))

# Dessin + extraction texte
result = orig.copy()
for idx,(x,y,w,h) in enumerate(expanded, 1):
    cv2.rectangle(result, (x,y), (x+w, y+h), (0,255,0), 2)

    # Zone sous le code-barres
    roi_y1 = y + h
    roi_y2 = min(h_img, y + h + int(h * 0.45) + 10)
    roi_x1 = x
    roi_x2 = x + w
    digits = ""
    if roi_y1 < roi_y2:
        roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        digits = extract_digits(roi)

        # Affichage sur image si trouvé
        if digits != "":
            cv2.putText(result, digits, (x, roi_y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Affichage dans la console
    print(f"Code-barres {idx}: position=({x},{y},{w},{h}), texte détecté='{digits}'")

# Sauvegarde
cv2.imwrite("sample_detected_console.jpg", result)
print("Fini → sample_detected_console.jpg généré")
