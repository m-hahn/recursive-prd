from paths import WIKIPEDIA_HOME
import random
 

itos_labels = []
stoi_labels = {}

sentenceIDs = {}

with open("/u/scr/mhahn/STIMULI/Staub2006.tsv", "w") as outFile:
  print("\t".join(["Sentence", "Item", "Condition", "Region", "Word", "NumInSent"]), file=outFile)
  with open("../stimuli/Staub_2006/staub2006_s_tokenized.tsv", "r") as inFile:
     text = [x.split("\t") for x in inFile.read().strip().split("\n")]
  header = text[0]
#  header = dict(zip(header, range(len(header))))
  text = text[1:]
  for line in text:
     #print(line)
     assert len(line) == len(header)
  chunk = []
  chunk_line_numbers = []
  regionCounter = 0
  lastSID = "NONE"
  for linenum, line in enumerate(text):
     line = dict(list(zip(header, line)))
#     if line["Round"] != "0":
 #     continue
     sentenceID = "_".join([line["Group"],line["Item"], line["Condition"], line["Round"]])
     if sentenceID != lastSID:
        regionCounter = 0
     lastSID = sentenceID
     word = line['Word'].split("_")
     for wn, w in enumerate(word):
        regionCounter += 1
        if line["Word"] == "or":
           region = "OR"
        elif line["Region"] == "other":
           region = f"other_{regionCounter}"
        else:
           region = line["Region"]+"_"+str(wn)
        print("\t".join([sentenceID, line["Item"], line["Condition"], region, w, str(regionCounter)]), file=outFile)
