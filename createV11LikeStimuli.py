import codecs

with codecs.open("/u/scr/mhahn/recursive-prd/VSLK_LCP/E3_DE_SPR/data/e3desprdata.txt", "r", "iso-8859-1") as inFile:
   text = [[y for y in x.strip().split(" ") if len(y) > 0] for x in inFile.read().strip().split("\n")]
header = ["subject","expt","item","condition","position","word","rt", "similarity", "grammaticality"] # according to VSLK_LCP/E1a_EN_SPR/analysis/rcode/processdata.R
header = dict(zip(header, range(len(header))))
chunk = []

items = {}

chunk_line_numbers = []
for linenum, line in enumerate(text):
   print(line)
   condition = line[header["condition"]]
   expt = line[header["expt"]]
   if condition not in ["a", "b"] :
      continue
   words = line[header['word']]
   if words == "Y": # remove yes-no question answers
      continue
   item = (int(line[header["item"]]), condition)
   position = int(line[header["position"]])-1
   assert position >= 0
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
   return ["DN1", "R1", "DN2", "R2", "DN3", "V3", "V2", "V1", "DN4"][x]

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
     print(item2) 
     print(x)
     print(items[item2])
     itemsResults[-1].append(([str(y) for y in items[item2][x][1:-1]+[region(x), c]]))

   itemsResults.append([])


char_vocab_path = "vocabularies/"+"german"+"-wiki-word-vocab-50000.txt"

with open(char_vocab_path, "r") as inFile:
     itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
stoi = dict([(itos[i],i) for i in range(len(itos))])


for item in itemsResults:
  sentence = [line[4] for line in item]
  regions = [line[7] for line in item]
  oov = [("-OOV" if any([y not in stoi for y in x.lower().replace(",", "").replace(".", "").split("_")]) else "") for x in sentence]
  if len(regions) == 0:
     continue
  assert len(sentence) == len(regions)
  other = item[0][:3] + [item[0][6], item[0][8]]
  print("\t".join(other)+"\t"+(" ".join([x+z+"@"+y for x, z, y in zip(sentence, oov, regions)])))

