import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
import functions as func 

img = shepp_logan_phantom()
n_angles = 128
sinogram = func.radon_transform(img,n_angles)
reco = func.iradon_transform(sinogram,n_angles)

plt.figure(figsize = (14,4))
plt.subplot(1,3,1)
plt.imshow(img,aspect = 'auto',cmap = 'jet')
plt.subplot(1,3,2)
plt.imshow(sinogram,aspect = 'auto',cmap = 'jet')
plt.subplot(1,3,3)
plt.imshow(reco,aspect = 'auto',cmap = 'jet')
plt.show()
