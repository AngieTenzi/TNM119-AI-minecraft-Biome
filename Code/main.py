import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage.color import rgb2hsv
from skimage import exposure
from skimage.feature import hog

imagePath = '../BiomeData/biome_1/'
image = mpimg.imread(imagePath + '/biome_1_354.jpg')

hsvImage = rgb2hsv(image)
hueImage = hsvImage[:,:,0]
satImage = hsvImage[:,:,1]
value_img = hsvImage[:, :, 2]

fd, hog_image = hog(
    image,
    orientations=8,
    pixels_per_cell=(16, 16),
    cells_per_block=(1, 1),
    visualize=True,
    channel_axis=-1,
)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

fig, (ax0, ax1, ax2, ax3,ax4) = plt.subplots(ncols=5, figsize=(8, 2))

ax0.imshow(image)
ax0.set_title("RGB image")
ax0.axis('off')
ax1.imshow(hueImage, cmap='hsv')
ax1.set_title("Hue channel")
ax1.axis('off')
ax2.imshow(value_img)
ax2.set_title("Value channel")
ax2.axis('off')
ax3.set_title("Saturation channel")
ax3.imshow(satImage)
ax3.axis('off')
ax4.axis('off')
ax4.imshow(hog_image, cmap=plt.cm.gray)
ax4.set_title('Histogram')

plt.show()

fig.tight_layout()

plt.show()