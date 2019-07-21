
with open("/u/scr/mhahn/recursive-prd/VSLK_LCP/E1_EN_SPR/data/e1_en_spr_data.txt", "r") as inFile:
   text = [x.split(" ") for x in inFile.read().strip().split("\n")]
header = ["subject","expt","item","condition","position","word","rt"] # according to VSLK_LCP/E1a_EN_SPR/analysis/rcode/processdata.R
header = dict(zip(header, range(len(header))))
chunk = []

items = {}

chunk_line_numbers = []
for linenum, line in enumerate(text):
   condition = line[header["condition"]]
   expt = line[header["expt"]]
   if condition not in ["a", "b"] or expt != "gug": 
      continue
   words = line[header['word']]
   if words == "Y": # remove yes-no question answers
      continue
   item = (int(line[header["item"]]), condition)
   position = int(line[header["position"]])
   if item not in items:
      items[item] = {}
   items[item][position] = line


def changeRegion(x, removed):
   x = x[:]
   region = int(x[3])
   if  region > removed:
       x[3] = region-1
   return x

def region(x):
   return ["D1", "N1", "R1", "D2", "N2", "R2", "D3", "N3", "V1", "V2", "V3", "D4", "N4"][x]

itemsResults = []

print(len(items))
for item in range(1, int(len(items)/2)+1):
 for c in ["a", "b"]:
   itemsResults.append([])
   v1 = 8
   v2 = 9
   item2 = (item, c)
   # full version 
   for x in range(len(items[item2])):
     
     itemsResults[-1].append(([str(y) for y in items[item2][x][1:-1]+[region(x), "full"]]))

   itemsResults.append([])

   # version with V1 missing
   for x in range(len(items[item2])):
     if x == 8:
        continue
     itemsResults[-1].append([str(y) for y in changeRegion(items[item2][x][1:-1], 8)+[region(x), "RemovedV1"]])

   itemsResults.append([])

   # version with V2 missing
   for x in range(len(items[item2])):
     if x == 9:
       continue
     itemsResults[-1].append([str(y) for y in changeRegion(items[item2][x][1:-1], 9)+[region(x), "RemovedV2"]])


import random
with open("stimuli/V11/English_EitherVerb.txt", "w") as outFile:
  print("\t".join(["expt","item","interference","position","word", "region", "grammatical", "iteration"]), file=outFile)
  for iteration in range(20):
     random.shuffle(itemsResults)
     for item in itemsResults:
       for line in item:
         print("\t".join(line+[str(iteration)]), file=outFile)

