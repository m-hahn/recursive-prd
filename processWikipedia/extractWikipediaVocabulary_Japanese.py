unigrams = {}

language = "japanese"

WIKIPEDIA_HOME = "/u/scr/mhahn/FAIR18/WIKIPEDIA/"+language+"/"
if True:
 pathIn  = WIKIPEDIA_HOME+""+language+"-train-tokenized.txt"
 pathOut = WIKIPEDIA_HOME+""+language+"-wiki-word-vocab.txt"

import random
with open(pathIn, "r") as inFile:
   counter = 0
   for line in inFile:
      counter += 1
      if counter % 1e5 == 0:
         print(counter)
      line = line.strip().replace("\ ", "").split(" ")
      for t in line:
        if len(t) == 1:
          continue
        try:
          word = t[:t.index("/")].lower()
          if len(word) == 0:
                continue
          unigrams[word] = unigrams.get(word, 0) + 1
        except ValueError:
           print(t)
 #     if random.random() > 0.99:
#          break
unigrams = sorted(list(unigrams.items()), key=lambda x:x[1],reverse=True)
with open(pathOut, "w") as outFile:
  for word, count in unigrams:
      print(f"{word}\t{count}", file=outFile)
      

