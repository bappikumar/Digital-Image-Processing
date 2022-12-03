from PIL import Image

img = Image.open('1.jpg')
print(img.format)
img.save('output.png')


