def  output(itemName, j, word, condition, outFile):
   print("\t".join([str(q) for q in [f"{itemName}_{condition}", itemName, condition, word[1], word[0], j]]), file=outFile)

from paths import WIKIPEDIA_HOME
import random
 

itos_labels = []
stoi_labels = {}

with open("/u/scr/mhahn/STIMULI/TaborHutchins.tsv", "w") as outFile:
  print("\t".join(["Sentence", "Item", "Condition", "Region", "Word", "NumInSent"]), file=outFile)
  with open("../stimuli/TaborHutchins2004/expt12.txt", "r") as inFile:
     text = [x.split("\t") for x in inFile.read().strip().split("\n")]
  for itemName, sentence in text:
    sentence = [[[q, ""] for q in  x.strip().split(" ")] for x in sentence.replace("(", "#").replace(")", "#").split("#")]
    for i in range(len(sentence[0])):
      sentence[0][i][1] = f"Pre_{i}"   
    for i in range(len(sentence[1])):
      sentence[1][i][1] = f"Inter1_{i}"   
    for i in range(len(sentence[2])):
      sentence[2][i][1] = f"Mid_{i}"   
    for i in range(len(sentence[3])):
      sentence[3][i][1] = f"Inter2_{i}"   
    for i in range(len(sentence[4])):
      sentence[4][i][1] = f"Final_{i}"    # Final_0 is the critical region
    for j, word in enumerate(sentence[0] + sentence[2] + sentence[4]):
       output(itemName, j, word, "Intransitive_Short", outFile)
    for j, word in enumerate(sentence[0] + sentence[1] + sentence[2] + sentence[4]):
       output(itemName, j, word, "Transitive_Short", outFile)
    for j, word in enumerate(sentence[0] + sentence[2] + sentence[3] + sentence[4]):
       output(itemName, j, word, "Intransitive_Long", outFile)
    for j, word in enumerate(sentence[0] + sentence[1] + sentence[2] + sentence[3] + sentence[4]):
       output(itemName, j, word, "Transitive_Long", outFile)

