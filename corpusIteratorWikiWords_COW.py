from paths import WIKIPEDIA_HOME
import random
 

def load(language, partition="train", removeMarkup=True):
  chunk = []
  if partition == "train":
     inFile = training_iterator()
  else:
     inFile = XZIterator("/u/scr/mhahn/CORPORA/COW/decow16bx/GERMAN_COW/"+partition+".txt.xz")
  for line in inFile:
  #    print(line)
      line = line.strip().split(" ")
      for word in line:
         chunk.append(word.lower())
         if len(chunk) > 1000000:
            yield chunk
            chunk = []
      chunk.append("<EOS>")
  yield chunk

import lzma
def XZIterator(path):
   with lzma.open(path, "rt", encoding="utf-8") as inFile:
      for line in inFile:
          yield line

def training_iterator():
   import tarfile
   BASE_PATH = "/u/scr/mhahn/CORPORA/COW/decow16bx/GERMAN_COW/"
   filehandles = []
   import lzma
   BASE_PATH_OUT = "/john0/scr1/mhahn/"
   for f in ["FIRST_TEN.tar.gz", "SECOND_TEN_John1.tar.gz",  "SECOND_TEN_John2.tar.gz",  "THIRD.tar.gz"]:
       print("Opening file "+f)
       tfile = tarfile.open(BASE_PATH+f, 'r|gz')
       for t in tfile:
           print(t, t.name)
           counter = 0
           f = tfile.extractfile(t)
           if f:
               for _ in range(1000):
                  _ = 0
               for _ in range(1000):
                  _ = 0
               try:
                 while True:
                   counter += 1
                   if counter % 1e5 == 0:
                      print(counter)
                   yield next(f).decode()
               except StopIteration:
                   print("Done processing file")

def training(language):
  return load(language, "train")
#   with open(WIKIPEDIA_HOME+""+language+"-train.txt", "r") as inFile:
#     data = inFile.read().strip().lower().split("\n")
#     print("Shuffling")
#     random.shuffle(data)
#     print("Finished shuffling")
#     return "".join(data)
def dev(language, removeMarkup=True):
  return load(language, "dev", removeMarkup=removeMarkup)

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


