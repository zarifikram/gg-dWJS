import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
data = pd.read_csv("gg-dWJS_samples.csv")

""" For amat files """
with open("data/binarized_mnist_test.amat") as f:
    lines = f.readlines()
data = np.array([[int(i) for i in line.split()] for line in lines])
# plt.axis('off')
# plt.tight_layout()
# for i in trange(data.shape[0]):
#     plt.imsave(f"figs/mnist_dwjs/{i}.png", data[i].reshape(28,28), cmap="gray", format="png")
#     plt.clf()

""" For samples.csv """
# plt.axis('off')
# plt.tight_layout()
# for i in trange(len(data)):
#     arr = data.iloc[i]['generated']
#     arr = np.array([0 if c == '-' else 1 for c in arr])
#     plt.imsave(f"figs/mnist_dgwjs/{i}.png", arr.reshape(28,28), cmap="gray", format="png")
#     plt.clf()

# # combine the 50 images into a mega image in a numpy array (new size 50x28x28)
row = 10
col = 40
# label = 8
# rand_indices = np.random.choice(list(range(1000*label, 1000*label+1000)), row*col, replace=False)
rand_indices = np.random.choice(len(data), row*col, replace=False)
mega_image = np.zeros((28*row, 28*col))
for i in range(row):
    for j in range(col):
        arr = data[rand_indices[i*col+j]]
        # arr = data.iloc[rand_indices[i*col+j]]['generated']
        # arr = np.array([0 if c == '-' else 1 for c in arr])
        mega_image[i*28:(i+1)*28, j*28:(j+1)*28] = arr.reshape(28, 28)

# show it
plt.imshow(mega_image, cmap="gray")
plt.axis('off')

plt.tight_layout()
plt.savefig(f"figs/test_samples.png", bbox_inches="tight", pad_inches = 0)