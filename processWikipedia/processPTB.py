# from https://www.asc.ohio-state.edu/demarneffe.1/LING5050/material/structured.html
header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]



from nltk.corpus import ptb
import os

def addTrees(sec, trees, outFile):
   secNum = ("" if sec >= 10 else "0") + str(sec)

   files = os.listdir("/u/scr/corpora/ldc/1999/LDC99T42/parsed/mrg/wsj/"+secNum)
   for name in files:
      for tree in ptb.parsed_sents("WSJ/"+secNum+"/"+name):
         leaves = " ".join([("(" if x == "-LRB-" else (")" if x == "-RRB-" else x.replace("\/", "/").replace("\*","*"))) for x in tree.leaves() if "*-" not in x and (not x.startswith("*")) and x not in ["0", "*U*", "*?*"]])
         if leaves not in deps: # only applies to one sentence in the training partition
            print(leaves)
            continue
         print(leaves, file=outFile)
         trees.append((tree, deps[leaves]))
          

def getPTB(partition, outFile):
   trees = []
   if partition == "train":
     sections = range(0, 19)
   elif partition in ["dev", "valid"]: # 19-21
     sections = range(19, 22)
   elif partition == "test": # 22-24
     sections = range(22, 25)
   for sec in sections:
      print(sec)
      addTrees(sec, trees, outFile)
   return trees

#print(getPTB("train"))
import os
import random
import sys



with open("/u/scr/mhahn/CORPORA/ptb-ud2/ptb-ud2.conllu", "r") as inFile:
   deps = inFile.read().strip().split("\n\n")
for i in range(len(deps)):
    words = " ".join([x.split("\t")[1] for x in deps[i].split("\n")   ])
    deps[i] = (words, deps[i])
deps = dict(deps)
print(len(deps))
print("Done reading deps")


for partition in ["train", "dev", "test"]:
  with open("/u/scr/mhahn/CORPORA/ptb-partitions/"+partition+".txt", "w") as outFile:
   getPTB(partition, outFile)


