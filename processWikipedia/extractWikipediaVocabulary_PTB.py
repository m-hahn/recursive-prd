unigrams = {}

language = "russian"

WIKIPEDIA_HOME = "/u/scr/mhahn/CORPORA/ptb-partitions/"
if True:
 pathIn  = WIKIPEDIA_HOME+"train.txt"
 pathOut = WIKIPEDIA_HOME+"ptb-word-vocab.txt"

import random
with open(pathIn, "r") as inFile:
   counter = 0
   for line in inFile:
      counter += 1
      if counter % 1e5 == 0:
         print(counter)
      line = line.strip().lower().split(" ")
      for word in line:
          unigrams[word] = unigrams.get(word, 0) + 1
unigrams = sorted(list(unigrams.items()), key=lambda x:x[1],reverse=True)
with open(pathOut, "w") as outFile:
  for word, count in unigrams:
      print(f"{word}\t{count}", file=outFile)
      

