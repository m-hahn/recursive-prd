from paths import WIKIPEDIA_HOME
import random
 

itos_labels = []
stoi_labels = {}

idToSentence = {}
idToSubject = {}

# open file
with open("/u/scr/mhahn/STIMULI/V11_E1_EN.tsv", "w") as outFile:
  print("\t".join(["Sentence", "Item", "Condition", "Region", "Word", "NumInSent"]), file=outFile)
  with open("/u/scr/mhahn/recursive-prd/VSLK_LCP/E1_EN_SPR/data/e1_en_spr_data.txt", "r") as inFile:
     text = [x.split(" ") for x in inFile.read().strip().split("\n")]
  header = ["subject","expt","item","condition","position","word","rt"] # according to VSLK_LCP/E1a_EN_SPR/analysis/rcode/processdata.R
  header = dict(zip(header, range(len(header))))
  for line in text:
     assert len(line) == len(header)
  chunk = []

  chunk_line_numbers = []
  sentIDLast = None
  for linenum, line in enumerate(text):
     word = line[header['word']]
     sentID0 = "_".join((line[header["expt"]], line[header["item"]], line[header["condition"]]))
     sentID = "_".join((line[header["subject"]], line[header["expt"]], line[header["item"]], line[header["condition"]]))
     if line[header["expt"]] not in ["gug"]:
       continue
     if line[header["condition"]] in ["practice", "filler"]:
       continue
#     print(line)
#     if line[header["correct"]] != "-":
 #       continue
     print(line) 
     if sentID != sentIDLast and sentID0 not in idToSubject:
         wordIndexInSent = 0
         idToSubject[sentID0] = line[header["subject"]]
     sentIDLast = sentID
     if idToSubject[sentID0] != line[header["subject"]]:
         continue
     if word[-1] in [".", "."]:
         word = [word[:-1], word[-1]]
     else:
         word = [word]
     for wordIndex, w in enumerate(word):
        #wordIndexInSent+=1
        region = line[header["position"]]
        if line[header["condition"]] in ["c", "d"]: # ungrammatical
          condition = "MissingVP"
          criticalRegion = "10"
        elif line[header["condition"]] in ["a", "b"]: # grammatical
          condition = "Full"
          criticalRegion = "11"
        else:
          assert False, line
        if region == criticalRegion:
             region="Critical"
        for q, word_ in enumerate(w.split("_")):
           print("\t".join([sentID0, line[header["expt"]]+"_"+line[header["item"]], condition+"_"+line[header["condition"]], region+"_"+str(q), word_, str(wordIndexInSent)]), file=outFile)
           wordIndexInSent+=1

