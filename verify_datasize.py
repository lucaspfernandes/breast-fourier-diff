from PIL import Image
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import numpy as np

img_files = sorted(glob.glob('./data/MAMA/*_CROP/*.tiff'))
print(f'We have {len(img_files)} images available.')
img_shapes = []

for item in img_files:
    img = Image.open(item)
    img_shapes.append(img.size)
img_shapes = np.array(img_shapes)

mean_x = img_shapes[:,0].mean()
mean_y = img_shapes[:,1].mean()

plt.figure()
plt.title('1rst dim histogram')
plt.xlabel('Shape')
plt.ylabel('#apparitions')
plt.axvline(mean_x, color='b', linestyle='dashed', linewidth=2, label='Mean')
plt.text(mean_x, -6, f'Mean: {mean_x:.2f}', color='b', ha='center')
plt.hist(img_shapes[:,0], color='r')
plt.legend()
plt.show()

plt.figure()
plt.title('2nd dim histogram')
plt.xlabel('Shape')
plt.ylabel('#apparitions')
plt.axvline(mean_y, color='b', linestyle='dashed', linewidth=2, label='Mean')
plt.text(mean_y, -6, f'Mean: {mean_y:.2f}', color='b', ha='center')
plt.hist(img_shapes[:,1], color='r')
plt.legend()
plt.show()

print(f'Mean 1rst dim = {mean_x}')
print(f'Mean 2nd dim = {mean_y}')