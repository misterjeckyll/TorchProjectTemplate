import matplotlib.pyplot as plt


# function to plot a grid of images with their labels
def plot_images(fig, image_index, row, col, img, title, textcolor='black'):
    sp = fig.add_subplot(row, col, image_index + 1)
    sp.axis('Off')
    sp.set_title(title, fontsize=16, color=textcolor)
    plt.imshow(img, interpolation=None)