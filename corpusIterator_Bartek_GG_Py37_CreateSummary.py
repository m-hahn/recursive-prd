forGeneration=False
tokenize=True
from paths import WIKIPEDIA_HOME
import random
 

itos_labels = []
stoi_labels = {}

idToSentence = {}
idToSubject = {}

# open file
with open("/u/scr/mhahn/STIMULI/BartekGG.tsv", "w") as outFile:
  print("\t".join(["Sentence", "Item", "Condition", "Region", "Word", "NumInSent"]), file=outFile)
  with open("../stimuli/BarteketalJEP2011data/gg-spr06-data.txt", "r") as inFile:
     text = [x.split("\t") for x in inFile.read().strip().split("\n")]
  #print(text)
  header = ["subj", "expt", "item", "condition", "roi", "word", "RT", "embedding", "intervention"] # bb-spr-dataprep.Rnw
  header = dict(zip(header, range(len(header))))
  for line in text:
     assert len(line) == len(header)
  chunk = []
  chunk_line_numbers = []
  sentIDLast = None
  for linenum, line in enumerate(text):
     print(line) 
     word = line[header['word']]
     sentID0 = "_".join((line[header["expt"]], line[header["item"]], line[header["condition"]]))
     sentID = "_".join((line[header["subj"]], line[header["expt"]], line[header["item"]], line[header["condition"]]))
     assert line[header["expt"]] == "E1"
     if line[header["expt"]] in ["filler", "gug", "E1collective"]:
       continue
     assert line[header["condition"]] in "abcdefg"
     if line[header["condition"]] in ["practice", "filler"]:
       continue
#     print(line)
     if sentID != sentIDLast and sentID0 not in idToSubject:
         wordIndexInSent = 0
         idToSubject[sentID0] = line[header["subj"]]
     sentIDLast = sentID
     if idToSubject[sentID0] != line[header["subj"]]:
         continue
     if word[-1] in [".", "."]:
         word = [word[:-1], word[-1]]
     else:
         word = [word]
     for wordIndex, w in enumerate(word):
        wordIndexInSent+=1
        region = line[header["roi"]]
        if line[header["expt"]] in ["E1", "E1collective"]:
           criticalRegion = str({"a" : 2, "b" : 5, "c" : 7, "d" : 5, "e" : 8, "f" : 10}[line[header["condition"]]])
       
        if region == criticalRegion:
             region="Critical"
        print("\t".join([sentID0, line[header["expt"]]+"_"+line[header["item"]], line[header["condition"]], region+"_"+str(wordIndex), w, str(wordIndexInSent)]), file=outFile)

