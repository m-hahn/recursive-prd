from paths import WIKIPEDIA_HOME
import random
 

itos_labels = []
stoi_labels = {}

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
    round_, item, condition, position, word, region = line #word = line[header['Word']].split("_")
    print("\t".join([item+"_"+condition+"_"+round_, item, condition, region+("_"+str(position) if region == "PostVerb" else ""), word, str(position)]), file=outFile)


