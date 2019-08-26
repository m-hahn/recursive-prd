from paths import WIKIPEDIA_HOME
import random
 

def load(language, partition="train", removeMarkup=True):
  if language == "italian":
    path = WIKIPEDIA_HOME+"/itwiki-"+partition+"-tagged.txt"
  elif language == "english":
    path = WIKIPEDIA_HOME+"/english-"+partition+"-tagged.txt"
  elif language == "german":
    path = WIKIPEDIA_HOME+""+language+"-"+partition+"-tagged.txt"
  else:
    path = WIKIPEDIA_HOME+"/WIKIPEDIA/"+language+"/"+language+"-"+partition+"-tagged.txt"
  chunk = []
  with open(path, "r") as inFile:
    for line in inFile:
      index = line.find("\t")
      if index == -1:
        if removeMarkup:
          continue
        else:
          index = len(line)-1
      word = line[:index]
      chunk.append(word.lower())
      if len(chunk) > 1000000:
      #   random.shuffle(chunk)
         yield chunk
         chunk = []
  yield chunk


# the first 1000 sentences per file serve as dev
import tarfile
BASE_PATH = "/u/scr/mhahn/CORPORA/COW/decow16bx/GERMAN_COW/"
filehandles = []
import lzma
BASE_PATH_OUT = "/john0/scr1/mhahn/"
with lzma.open(BASE_PATH_OUT+"dev.txt.xz", "wb") as dev:
  with lzma.open(BASE_PATH_OUT+"test.txt.xz", "wb") as test:
    with lzma.open(BASE_PATH_OUT+"train.txt.xz", "wb") as train:
       for f in ["FIRST_TEN.tar.gz", "SECOND_TEN_John1.tar.gz",  "SECOND_TEN_John2.tar.gz",  "THIRD.tar.gz"]:
          tfile = tarfile.open(BASE_PATH+f, 'r|gz')
          for t in tfile:
              print(t, t.name)
              counter = 0
              f = tfile.extractfile(t)
              if f:
                  for _ in range(1000):
                     dev.write(next(f))
                  for _ in range(1000):
                     test.write(next(f))
                  try:
                     while True:
                        counter += 1
                        if counter % 1e5 == 0:
                           print(counter)
                        train.write(next(f))
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


