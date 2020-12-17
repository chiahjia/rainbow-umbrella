import numpy as np
from sklearn.decomposition import NMF
import cv2
import matplotlib.pyplot as plt

filenames = ['people2.jpg', 'people3.jpg']
base = cv2.cvtColor(cv2.imread(filenames[0]), cv2.COLOR_BGR2GRAY)
test = cv2.cvtColor(cv2.imread(filenames[1]), cv2.COLOR_BGR2GRAY)

model = NMF(n_components=10, init='random', random_state=0, max_iter=1000)

W = model.fit_transform(base)
H = model.components_
W2 = model.transform(test)
test_after = np.matmul(W2, H)
base_after = np.matmul(W, H)

plt.subplot(2, 2, 1)
plt.imshow(base, cmap='gray')
plt.subplot(2, 2, 2)
plt.imshow(base_after, cmap='gray')

plt.subplot(2, 2, 3)
plt.imshow(test, cmap='gray')
plt.subplot(2, 2, 4)
plt.imshow(test_after, cmap='gray')

plt.show()
