from paths import WIKIPEDIA_HOME
import random



def process(item, condition, sentence):
  print(sentence, item, condition)
  for index, word in enumerate(sentence):
    print("\t".join([str(q) for q in [f"{item}_{condition}", item, condition, word[1], word[0], index]]), file=outFile)

itos_labels = []
stoi_labels = {}

idToSentence = {}
idToSubject = {}

with open("/u/scr/mhahn/STIMULI/VanDyke_Lewis_2003.tsv", "w") as outFile:
  print("\t".join(["Sentence", "Item", "Condition", "Region", "Word", "NumInSent"]), file=outFile)
  items = []
  with open("../stimuli/VanDyke_Lewis_2003/items.txt", "r") as inFile:
    data = inFile.read().strip().split("\n")
    print(len(data))
    for item in data:
      print(item)
      itemID = int(item[:item.index(".")])
      item = item[item.index(" ")+1:]
      print(itemID, item)
      x1, x2 = item.split("(")
      x2, x3 = x2.split(")")
      x3, x4 = x3.split("]")
      x3a, x3b, x3c = x3.split("%")
      x3a = x3a.strip().lstrip("[")
      x1 = [(word, f"R1_{i}") for i, word in enumerate(x1.strip().split(" "))]
      x2 = [(word, f"R2_{i}") for i, word in enumerate(x2.strip().split(" "))]
      x3a = [(word, f"R3_{i}") for i, word in enumerate(x3a.strip().split(" "))]
      x3b = [(word, f"R3_{i}") for i, word in enumerate(x3b.strip().split(" "))]
      x3c = [(word, f"R3_{i}") for i, word in enumerate(x3c.strip().split(" "))]
      x4 = [(word, f"R4_{i}") for i, word in enumerate(x4.replace(".", " .").replace("  ", " ").strip().split(" "))]
  
  
      
      print(x1, x2, x3a, x3b, x3c, x4)
      process(f"{itemID}", "that_1", x1+x2+x3a+x4)
      process(f"{itemID}", "that_2", x1+x2+x3b+x4)
      process(f"{itemID}", "that_3", x1+x2+x3c+x4)
  
      process(f"{itemID}", "nothat_1", x1+x3a+x4)
      process(f"{itemID}", "nothat_2", x1+x3b+x4)
      process(f"{itemID}", "nothat_3", x1+x3c+x4)

