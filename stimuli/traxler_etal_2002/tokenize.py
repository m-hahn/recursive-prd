with open("expt1.txt", "r") as inFile:
  lines = inFile.read().strip().split("\n")

def region(cond, pos):
   src = ["d0", "n0", "c", "v0", "d1", "n1", "v1"]
   orc = ["d0", "n0", "c", "d1", "n1", "v0", "v1"]
   if pos >= len(src):
      return "post"
   if cond == "SRC":
      return src[pos]
   elif cond == "ORC":
      return orc[pos]

itemsInConditions = []

for item, line in enumerate(lines):
    line = line.lower().replace("/that ", "/").replace("/", " ").replace(",", ", ").replace(".", " .").split(" ")
    version1 = line[:6]+line[9:]
    version2 = line[:3]+line[6:]
#    print(list(zip(line, range(len(line)))))
    v1=(" ".join(version1))
    v2=(" ".join(version2))
 #   print(v1)
  #  print(v2)
    assert set(version1) == set(version2)
    assert len(v1) == len(v2)
    assert version1[2] == "that"
    assert version1[4] == "the"
    assert version2[2] == "that"
    assert version2[3] == "the"
    for condition, sent in [("SRC", version1), ("ORC", version2)]:
       itemsInConditions.append([])
       for position, word in enumerate(sent):
          itemsInConditions[-1].append((item, position, word, condition, region(condition, position)))

import random
random.shuffle(itemsInConditions)

with open("expt1-tokenized.tsv", "w") as outFile:
  print("\t".join(["Item", "Position", "Word", "Condition", "Region"]), file=outFile)
  for line in itemsInConditions:
     for word in line:
         print("\t".join([str(x) for x in word]), file=outFile)

