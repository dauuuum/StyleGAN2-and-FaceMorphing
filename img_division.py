from PIL import Image

imgNum  = 1
x, y = 0, 0
w, h = 1024, 1024
img = Image.open('./train/fakes000060.png')
for i in range(0,20):
  try:
    area = (x,y,w,h)
    cropped = img.crop(area)
    cropped.save("Test_{}.png".format(imgNum),'PNG')
    imgNum += 1
    x += 1024
    y += 1024
    w += 1024
    h += 1024
  except:
    continue
