import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class BarcodeDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Détecteur de Codes-Barres")
        self.root.geometry("1000x700")
        
        self.img = None
        self.img_display = None
        self.bd = cv2.barcode.BarcodeDetector()
        
        # Frame pour les boutons
        btn_frame = tk.Frame(root, bg="#f0f0f0", pady=10)
        btn_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Bouton pour importer une image
        self.btn_import = tk.Button(
            btn_frame,
            text="📁 Importer une image",
            command=self.import_image,
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.btn_import.pack(side=tk.LEFT, padx=10)
        
        # Bouton pour détecter les codes-barres
        self.btn_detect = tk.Button(
            btn_frame,
            text="🔍 Détecter les codes-barres",
            command=self.detect_barcode,
            font=("Arial", 12, "bold"),
            bg="#2196F3",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.btn_detect.pack(side=tk.LEFT, padx=10)
        
        # Bouton pour sauvegarder
        self.btn_save = tk.Button(
            btn_frame,
            text="💾 Sauvegarder",
            command=self.save_image,
            font=("Arial", 12, "bold"),
            bg="#FF9800",
            fg="white",
            padx=20,
            pady=10,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.btn_save.pack(side=tk.LEFT, padx=10)
        
        # Label pour afficher les informations
        self.info_label = tk.Label(
            root,
            text="Aucune image chargée",
            font=("Arial", 10),
            bg="#e0e0e0",
            pady=5
        )
        self.info_label.pack(side=tk.TOP, fill=tk.X)
        
        # Frame pour l'image avec scrollbar
        canvas_frame = tk.Frame(root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas pour afficher l'image
        self.canvas = tk.Canvas(canvas_frame, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar = tk.Scrollbar(root, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X, before=canvas_frame)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
    
    def import_image(self):
        """Importer une image depuis le système de fichiers"""
        file_path = filedialog.askopenfilename(
            title="Sélectionner une image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Tous les fichiers", "*.*")
            ]
        )
        
        if file_path:
            self.img = cv2.imread(file_path)
            if self.img is None:
                messagebox.showerror("Erreur", "Impossible de charger l'image")
                return
            
            self.img_path = file_path
            self.display_image(self.img)
            self.btn_detect.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.DISABLED)
            self.info_label.config(text=f"Image chargée : {file_path}")
    
    def detect_barcode(self):
        """Détecter les codes-barres sur l'image"""
        if self.img is None:
            messagebox.showwarning("Attention", "Veuillez d'abord importer une image")
            return
        
        # Copie de l'image pour le traitement
        img_result = self.img.copy()
        
        # Détection
        if hasattr(self.bd, "detectAndDecodeWithType"):
            retval, decoded_info, decoded_type, points = self.bd.detectAndDecodeWithType(img_result)
        else:
            retval, decoded_info, decoded_type, points = self.bd.detectAndDecode(img_result)
        
        if not retval or points is None or len(points) == 0:
            messagebox.showinfo("Résultat", "Aucun code-barres détecté sur cette image")
            self.info_label.config(text="Aucun code-barres détecté")
            return
        
        # Traitement des points
        points = np.array(points, dtype=np.int32)
        if points.ndim == 2:
            points = points.reshape(1, 4, 2)
        
        # Dessiner les polygones
        img_result = cv2.polylines(img_result, points, True, (0, 255, 0), 10)
        
        # Afficher les informations détectées
        info_text = f"{len(decoded_info)} code(s)-barres détecté(s) : "
        
        for i, (text, pts, typ) in enumerate(zip(decoded_info, points, decoded_type)):
            x, y = pts[1]  # topLeft
            label = f"{text} ({typ})"
            img_result = cv2.putText(
                img_result,
                label,
                (int(x), int(y) + 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 0, 255),
                5,
                cv2.LINE_AA
            )
            info_text += f"{text} ({typ})" + (", " if i < len(decoded_info) - 1 else "")
        
        self.img_display = img_result
        self.display_image(img_result)
        self.btn_save.config(state=tk.NORMAL)
        self.info_label.config(text=info_text)
        
        messagebox.showinfo("Succès", f"{len(decoded_info)} code(s)-barres détecté(s) !")
    
    def display_image(self, img):
        """Afficher l'image sur le canvas"""
        # Convertir de BGR à RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Redimensionner si l'image est trop grande
        h, w = img_rgb.shape[:2]
        max_size = 800
        if h > max_size or w > max_size:
            scale = min(max_size / h, max_size / w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h))
        
        # Convertir en ImageTk
        img_pil = Image.fromarray(img_rgb)
        self.photo = ImageTk.PhotoImage(img_pil)
        
        # Afficher sur le canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
    
    def save_image(self):
        """Sauvegarder l'image traitée"""
        if self.img_display is None:
            messagebox.showwarning("Attention", "Aucune image traitée à sauvegarder")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[
                ("JPEG", "*.jpg"),
                ("PNG", "*.png"),
                ("Tous les fichiers", "*.*")
            ]
        )
        
        if file_path:
            success = cv2.imwrite(file_path, self.img_display)
            if success:
                messagebox.showinfo("Succès", f"Image sauvegardée : {file_path}")
            else:
                messagebox.showerror("Erreur", "Impossible de sauvegarder l'image")

# Lancement de l'application
if __name__ == "__main__":
    root = tk.Tk()
    app = BarcodeDetectorApp(root)
    root.mainloop()