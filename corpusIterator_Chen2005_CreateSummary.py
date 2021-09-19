forGeneration=False
tokenize=True
from paths import WIKIPEDIA_HOME
import random
 

itos_labels = []
stoi_labels = {}

idToSentence = {}
idToSubject = {}
processed = set()

# open file
with open("/u/scr/mhahn/STIMULI/Chen2005.tsv", "w") as outFile:
  print("\t".join(["Sentence", "Item", "Condition", "Region", "Word", "NumInSent"]), file=outFile)
  with open("../stimuli/Chen_etal_2005/expt1-tokenized.tsv", "r") as inFile:
     text = [x.split("\t") for x in inFile.read().strip().split("\n")]
  #print(text)
  header = text[0]
  header = dict(zip(header, range(len(header))))
  text = text[1:]
  for line in text:
     assert len(line) == len(header)
  sentIDLast = None
  for linenum, line in enumerate(text):
     print(line) 
     word = line[header["Word"]]
     sentID0 = "_".join((line[header["Item"]], line[header["Condition"]]))
     sentID = "_".join((line[header["Round"]], line[header["Item"]], line[header["Condition"]]))
     if sentID != sentIDLast and sentID0 not in idToSubject:
         wordIndexInSent = 0
         idToSubject[sentID0] = line[header["Round"]]
     sentIDLast = sentID
     if idToSubject[sentID0] != line[header["Round"]]:
         continue
     word = word.split("_")
     for wordIndex, w in enumerate(word):
        wordIndexInSent+=1
        region = line[header["Region"]]
        print("\t".join([sentID0, line[header["Item"]], line[header["Condition"]], region+"_"+str(wordIndex), w, str(wordIndexInSent)]), file=outFile)

