frequencies = {}
import html
with open("/john2/scr1/mhahn/ENCOW_EXCEPT2/encow16ax.wp.tsv", "r") as inFile:
#with open("/john0/scr1/mhahn/decow16bx.wp.tsv", "r") as inFile:
   for line in inFile:
      line = line.strip().split("\t")
      word = html.unescape(line[0]).lower()
      frequencies[word] = int(line[2]) + frequencies.get(word, 0)
      if len(frequencies) > 1e6:
         break
frequencies = list(frequencies.items())
import random
random.shuffle(frequencies)
frequencies = sorted(frequencies, key=lambda x:-x[1])
with open("/u/scr/mhahn/CORPORA/COW/decow16bx/english-cow-word-vocab.txt", "w") as outFile:
  for word, freq in frequencies:
       print(word+"\t"+str(freq), file=outFile)
