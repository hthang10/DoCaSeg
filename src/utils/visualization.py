import numpy as np
import matplotlib.pyplot as plt


def de_normalize(
    img,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
):
    result = img * std + mean
    result = np.clip(result, 0.0, 1.0)
    return result


def show_image(img):
    plt.imshow(de_normalize(img))
    plt.axis("off")
    plt.show()