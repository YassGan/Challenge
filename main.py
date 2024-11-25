import cv2
import numpy as np
from matplotlib import pyplot as plt

# Charger l'image
img = cv2.imread('data/non_annotated/1.JPG')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Appliquer un filtre gaussien pour lisser l'image
gaussian_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

# Appliquer CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(8, 8))
clahe_img = clahe.apply(gaussian_img)

# Appliquer la segmentation
_, threshold_img = cv2.threshold(clahe_img, 135, 255, cv2.THRESH_BINARY)

# Trouver les contours des régions segmentées
contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

regions_features = []

# Filtrer et extraire les caractéristiques des régions détectées
for contour in contours:
    # Calculer l'aire de la région
    area = cv2.contourArea(contour)
    
    # Ignorer les régions trop petites 
    if area < 150:  
        continue

    # Calculer la boîte englobante (bounding box)
    x, y, w, h = cv2.boundingRect(contour)
    
    # Extraire la région (ROI)
    roi = gray_img[y:y+h, x:x+w]
    

    # Dessiner le contour et la boîte englobante sur l'image originale (pour visualisation)
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)


# Sauvegarder l'image avec regions détectées
cv2.imwrite('output/output_image.JPG', img)

# Afficher les résultats
plt.figure(figsize=(10, 7))

# Image originale avec les contours et boîtes englobantes
plt.title("Contours détectés")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()




