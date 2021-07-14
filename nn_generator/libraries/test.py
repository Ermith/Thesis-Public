def func(what):
    '''what is this'''
    print(what)

func.__doc__ = "Simple string to print something"


help(func)

'''
colors = init_cmaps()

images = []

images.append(cv2.imread("raw_data/cz/heights.pgm", cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("raw_data/cz/roads.pgm", cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("raw_data/cz/rivers.pgm", cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("raw_data/cz/buildings.pgm", cv2.IMREAD_GRAYSCALE))

for i in range(len(images)):
    images[i] = cutout(images[i], 100, 100, 300, 300)

save_all(images, colors, "image.png")
'''