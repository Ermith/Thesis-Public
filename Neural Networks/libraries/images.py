from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

def register_colormap(cmap):
    ncolors = 256
    color_array = plt.get_cmap(cmap)(range(ncolors))
    color_array[:,-1] = np.linspace(0.0,1.0,ncolors)
    map_object = LinearSegmentedColormap.from_list(name=f'{cmap}_alpha',colors=color_array)

    plt.register_cmap(cmap=map_object)

def init_cmaps():
    register_colormap("Blues")
    register_colormap("Greens")
    register_colormap("turbo")

    return ['copper', 'Blues_alpha', 'Greens_alpha', 'turbo_alpha']

def save_as_pgm(matrix, file_name):
    _max = np.max(matrix)
    with open(file_name + ".pgm", "w") as f:
        f.write("P2\n") # P2 -> PGM mark
        f.write(f"{matrix.shape[1]}\n")
        f.write(f"{matrix.shape[0]}\n")
        f.write(f"{_max}")

        for row in range(matrix.shape[0]):
            separator = ""
            f.write("\n")
            for col in range(matrix.shape[1]):
                f.write(separator)
                f.write(str(matrix[row, col]))

                separator = " "

def save_image(img, name):
    plt.gca().set_axis_off()
    plt.imshow(img)
    plt.savefig(name, bbox_inches='tight', pad_inches=0)

def show_image(img):
    plt.imshow(img)
    plt.show()

def show_3d(img, color):
    x, y = np.meshgrid(range(img.shape[0]), range(img.shape[1]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, img, color=color)
    plt.show()

def show_all(images, colors):
    for i in range(len(images)):
        plt.imshow(images[i], cmap=colors[i])

    plt.show()

def save_all(images, colors, name):
    plt.gca().set_axis_off()
    for i in range(len(images)):
        plt.imshow(images[i], cmap=colors[i])
    
    plt.savefig(name, bbox_inches='tight', pad_inches=0)