frequencies = {}
import html
with open("/john2/scr1/mhahn/ESCOW/escow14ax.freq0.w.tsv", "r") as inFile:
   next(inFile) # header
   for line in inFile:
      line = line.strip().split("\t")
      word = html.unescape(line[6]).lower()
      frequencies[word] = int(line[0]) + frequencies.get(word, 0)
      if len(frequencies) > 1e6:
         break
frequencies = list(frequencies.items())
import random
random.shuffle(frequencies)
frequencies = sorted(frequencies, key=lambda x:-x[1])
with open("/u/scr/mhahn/CORPORA/COW/decow16bx/spanish-cow-word-vocab.txt", "w") as outFile:
  for word, freq in frequencies:
       print(word+"\t"+str(freq), file=outFile)
