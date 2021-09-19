from paths import WIKIPEDIA_HOME
import random
position = 0 

itos_labels = []
stoi_labels = {}
lastItemID = None

itemsDone = set()
# open file
with open("/u/scr/mhahn/STIMULI/Staub_2016.tsv", "w") as outFile:
  print("\t".join(["Sentence", "Item", "Condition", "Region", "Word", "NumInSent"]), file=outFile)
  with open("../stimuli/Staub_2016/stims-tokenized.tsv", "r") as inFile:
     text = [x.split("\t") for x in inFile.read().strip().split("\n")]
  header = text[0]
  header = dict(zip(header, range(len(header))))
  text = text[1:]
  for line in text:
     #print(line)
     assert len(line) == len(header)
  chunk = []
  chunk_line_numbers = []
  for linenum, line in enumerate(text):
    round_, item, condition, pos_, word, region = line #word = line[header['Word']].split("_")
    itemID = item+"_"+condition+"_"+round_
    if itemID != lastItemID:
        position = 0
        if item+"_"+condition in itemsDone:
           continue
        else:
           itemsDone.add(item+"_"+condition)
        assert int(pos_) == 0, line
    lastItemID = itemID
    for q, word_ in enumerate(word.split("_")):
       print("\t".join([item+"_"+condition+"_"+round_, item, condition, region+("_"+str(position) if region == "PostVerb" else "")+("_"+str(q) if q  > 0 else ""), word_, str(position)]), file=outFile)
       position += 1


