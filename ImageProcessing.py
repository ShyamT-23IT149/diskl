import cv2
import matplotlib.pyplot as plt


img = cv2.imread("D:\SEM 6\DS\dsl\muffin\img_0_50.jpg")

img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray,100,200)

plt.imshow(edges)
plt.title("A")
plt.axis("off")
plt.show()