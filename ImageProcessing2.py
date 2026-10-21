import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

path = r"D:\SEM 6\DS\dsl 2\muffin\img_0_50.jpg"

try:
    pil_img = Image.open(path).convert("RGB")
    img_rgb = np.array(pil_img)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    plt.imshow(edges, cmap="gray")
    plt.title("Edge Detection")
    plt.axis("off")
    plt.show()

except Exception as e:
    print("Image loading error:", e)

//To verify image size//

import os
import cv2

path = r"D:\SEM 6\DS\dsl 2\muffin\img_0_50.jpg"

print("File exists:", os.path.exists(path))
print("File size:", os.path.getsize(path), "bytes")

img = cv2.imread(path)

if img is None:
    print("OpenCV cannot read this image.")
else:
    print("Image loaded successfully!")
    print("Image dimensions:", img.shape)
//
