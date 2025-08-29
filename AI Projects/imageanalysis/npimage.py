import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image 
import requests
from io import BytesIO

def load_img(url):
    response=requests.get(url)
    return Image.open(BytesIO(response.content))

# Elephant image URL
result = "https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg"

# Load elephant image
resilt2 = load_img(result)

plt.figure(figsize=(6,4))
plt.imshow(resilt2)
plt.title("Elephant123")
plt.axis("off")
plt.show()

resultarray=np.array(resilt2)
print('elephantimpage',resultarray.reshape)
elephant_gray = resilt2.convert("L")

# Display grayscale image
plt.figure(figsize=(6, 4))
plt.imshow(elephant_gray, cmap="gray")
plt.title("Elephant (Grayscale)")
plt.axis("off")
plt.show()


