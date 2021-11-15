from PIL import Image

img = Image.open('./train/fakes000060.png')

imgNum  = 1
x, y = 0, 0
w, h = 1024, 1024
for i in range(1,8):
  try:
      area = (x,y,w,h)
      cropped = img.crop(area)
      cropped.save("./result/Test_{}.png".format(imgNum),'PNG')
      imgNum += 1
      x += 1024
      w += 1024
  except:
    continue

imgNum  = 8
x, y = 0, 1024
w, h = 1024, 2048
for i in range(1,8):
  try:
      area = (x,y,w,h)
      cropped = img.crop(area)
      cropped.save("./result/Test_{}.png".format(imgNum),'PNG')
      imgNum += 1
      x += 1024
      w += 1024
  except:
    continue

imgNum  = 15
x, y = 0, 2048
w, h = 1024, 3072
for i in range(1,8):
  try:
      area = (x,y,w,h)
      cropped = img.crop(area)
      cropped.save("./result/Test_{}.png".format(imgNum),'PNG')
      imgNum += 1
      x += 1024
      w += 1024
  except:
    continue

imgNum  = 22
x, y = 0, 3072
w, h = 1024, 4096
for i in range(1,8):
  try:
      area = (x,y,w,h)
      cropped = img.crop(area)
      cropped.save("./result/Test_{}.png".format(imgNum),'PNG')
      imgNum += 1
      x += 1024
      w += 1024
  except:
    continue