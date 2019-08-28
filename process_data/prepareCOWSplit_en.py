import random
counter = 0
# the first 1000 sentences per file serve as dev
import tarfile
BASE_PATH = "/john2/scr1/mhahn/ENCOW_EXCEPT2/"
import os
files = [x for x in os.listdir(BASE_PATH) if x.endswith(".shuf.txt")]
filehandles = []
import lzma
import html
BASE_PATH_OUT = "/john2/scr1/mhahn/ENCOW_EXCEPT2/"
with lzma.open(BASE_PATH_OUT+"dev.txt.xz", "wb") as dev:
  with lzma.open(BASE_PATH_OUT+"test.txt.xz", "wb") as test:
    with lzma.open(BASE_PATH_OUT+"train.txt.xz", "wb") as train:
       for f in files:
          print(f)
          with open(BASE_PATH+f, 'r') as tfile:
             for _ in range(1000):
                dev.write(html.unescape(next(tfile)).encode())
             for _ in range(1000):
                test.write(html.unescape(next(tfile)).encode())
             try:
                while True:
                   counter += 1
                   if counter % 1e5 == 0:
                      print(counter)
                   train.write(html.unescape(next(tfile)).encode())
             except StopIteration:
                   print("Done")
       #       if "file3" in t.name: 
        #          f = tfile.extractfile(t)
         #         if f:
          #            print(len(f.read()))
   
quit()
if True:
  tar = tarfile.open(BASE_PATH+f, "r:gz")
  print(f)
  for member in tar.getmembers():
    print(member)
    f = tar.extractfile(member)
    if f is not None:
      filehandles.append(f)
      print(filehandles)
  #    content = f.read()

def training(language):
  return load(language, "train")
#   with open(WIKIPEDIA_HOME+""+language+"-train.txt", "r") as inFile:
#     data = inFile.read().strip().lower().split("\n")
#     print("Shuffling")
#     random.shuffle(data)
#     print("Finished shuffling")
#     return "".join(data)
def dev(language, removeMarkup=True):
  return load(language, "valid", removeMarkup=removeMarkup)

def test(language, removeMarkup=True):
  return load(language, "test", removeMarkup=removeMarkup)

#   with open(WIKIPEDIA_HOME+""+language+"-valid.txt", "r") as inFile:
#     data = inFile.read().strip().lower().split("\n")
#     print("Shuffling")
#     random.shuffle(data)
#     print("Finished shuffling")
#     return "".join(data)
#

#     for line in data:
#        yield line


