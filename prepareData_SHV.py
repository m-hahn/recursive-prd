
import stanfordnlp
#stanfordnlp.download('hi')
nlp = stanfordnlp.Pipeline(processors = "tokenize,lemma,pos", model_path="/u/scr/mhahn/software/stanfordnlp_resources/fa_hdtb_models/", lang="fa", use_gpu=True)

from parsivar import Normalizer
my_normalizer = Normalizer(statistical_space_correction=True)

language = "farsi"

char_vocab_path = "/u/scr/mhahn/FAIR18/WIKIPEDIA/farsi/farsi-wiki-word-vocab.txt"
#char_vocab_path = "vocabularies/"+language.lower()+"-wiki-word-vocab-50000.txt"

print("Loading vocab")

with open(char_vocab_path, "r") as inFile:
     itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")]
stoi = dict([(itos[i],i) for i in range(len(itos))])

print("Done loading vocab")

with open("vocabularies/char-vocab-wiki-"+language, "r") as inFile:
     itos_chars = [x for x in inFile.read().strip().split("\n")]
stoi_chars = dict([(itos_chars[i],i) for i in range(len(itos_chars))])


items = []
with open("stimuli/safavi_etal_2016_persian/items_merged_beforeEdit_REGIONS.txt", "r") as inFile:
   typ = "1 1 1"
   for lineNumber, line in enumerate(inFile):
      if lineNumber == 0 or lineNumber == 36*5+1:
          print(line)
      part = (1 if  lineNumber < 36*5+1 else 2)
      item = int((lineNumber-1)/5)+1 if part == 1 else int((lineNumber-1-36*5-1)/5)+1

      condition =  int((lineNumber-1) % 5) if part == 1 else int((lineNumber-1-36*5-1) % 5)
      if condition == 1:
         items.append([])
      if condition >= 1 and lineNumber != 0 and lineNumber != 36*5+1:
         items[-1].append((part, item, condition, line.strip()))



tokenizedTrials = []

for item in items:
   print(item[0][:3])

   sent2TargetStart = item[1][3].index("@")+1
   sent2TargetEnd = item[1][3].index(" ", sent2TargetStart)
   target2 = item[1][3][sent2TargetStart:sent2TargetEnd]

   sent4TargetStart = item[3][3].index("@")+1
   sent4TargetEnd = item[3][3].index(" ", sent4TargetStart)
   target4 = item[3][3][sent4TargetStart:sent4TargetEnd]

   part = item[1][0]
   itemID = item[1][1]


   for version in item:
         condition = version[2]
         line = version[3].strip()+"."
         line = line.replace("@", "")
         doc = nlp(line)
         text_tokenized = []
         position = 0
         hasFoundTarget = False
         tokenizedTrials.append([])
         for sent in doc.sentences:
           for wrd in sent.words:
              position += 1
              if not hasFoundTarget and (((wrd.text == target2) and condition in [1,2]) or ((wrd.text == target4) and condition in [3,4])):
                  hasFoundTarget = True
                  region = "verb"
              else:
                  region = "other"
              tokenizedTrials[-1].append(("\t".join([str(x) for x in [position, wrd.text, wrd.lemma, wrd.pos, part, itemID, condition, region]])))
         assert hasFoundTarget, (target2 , target4)

import random
with open("stimuli/safavi_etal_2016_persian/items_merged_tokenized.txt", "w") as outFile:
   print("\t".join([str(z) for z in ["Position", "Word", "Lemma", "POS", "Part", "Item", "Condition", "Region", "Round"]]), file=outFile)
   for ROUND in range(5):
     print(ROUND)
     random.shuffle(tokenizedTrials)
     for trial in tokenizedTrials:
        for line in trial:
          print(line+"\t"+str(ROUND), file=outFile)


#      print(part, item, condition)
#      print(line)
#      continue
#      if len(line) > 3:
#         
         
#         i=0

#
#
