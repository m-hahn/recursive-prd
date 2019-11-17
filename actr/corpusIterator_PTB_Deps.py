# from https://www.asc.ohio-state.edu/demarneffe.1/LING5050/material/structured.html
header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]


import os

def addTrees(trees, partition):
   with open("/u/scr/mhahn/CORPORA/ptb-partitions/"+partition+".txt", "r") as inFile:
      for line in inFile:
         line.strip()
         if line not in deps: # only applies to one sentence in the training partition
            print(line)
            continue
         trees.append((tree, deps[line]))

def getPTB(partition):
   trees = []
   addTrees(trees, partition)
   return trees

import os
import random
import sys



#with open("/u/scr/mhahn/CORPORA/ptb-ud2/ptb-ud2.conllu", "r") as inFile:
#   deps = inFile.read().strip().split("\n\n")
#for i in range(len(deps)):
#    words = " ".join([x.split("\t")[1] for x in deps[i].split("\n")   ])
#    deps[i] = (words, deps[i])
#deps = dict(deps)
#print(len(deps))
#print("Done reading deps")
#

def load(language, partition="train", removeMarkup=True):
  chunk = []
  with open("/u/scr/mhahn/CORPORA/ptb-partitions/"+partition+".txt", "r") as inFile:  
    for line in inFile:
      for word in line.strip().split(" "):
        chunk.append(word.lower())
        if len(chunk) > 1000000:
           yield chunk
           chunk = []
  yield chunk

def training(language):
  return load(language, "train")
def dev(language, removeMarkup=True):
  return load(language, "valid", removeMarkup=removeMarkup)
def test(language, removeMarkup=True):
  return load(language, "test", removeMarkup=removeMarkup)




class CorpusIterator_PTB():
   def __init__(self, language, partition="train"):
      data = getPTB(partition)
#      if shuffleData:
#       if shuffleDataSeed is None:
#         random.shuffle(data)
#       else:
#         random.Random(shuffleDataSeed).shuffle(data)

      self.data = data
      self.partition = partition
      self.language = language
      assert len(data) > 0, (language, partition)
   def permute(self):
      random.shuffle(self.data)
   def length(self):
      return len(self.data)
   def getSentence(self, index):
      result = self.processSentence(self.data[index])
      return result
   def iterator(self):
     for sentence in self.data:
        yield self.processSentence(sentence)
   def processSentence(self, sentenceAndTree):
        tree, sentence = sentenceAndTree
        sentence = map(lambda x:x.split("\t"), sentence.split("\n"))
        result = []
        for i in range(len(sentence)):
#           print sentence[i]
           if sentence[i][0].startswith("#"):
              continue
           if "-" in sentence[i][0]: # if it is NUM-NUM
              continue
           if "." in sentence[i][0]:
              continue
           sentence[i] = dict([(y, sentence[i][x]) for x, y in enumerate(header)])
           sentence[i]["head"] = int(sentence[i]["head"])
           sentence[i]["index"] = int(sentence[i]["index"])
           sentence[i]["word"] = sentence[i]["word"].lower()
           if self.language == "Thai-Adap":
              assert sentence[i]["lemma"] == "_"
              sentence[i]["lemma"] = sentence[i]["word"]
           if "ISWOC" in self.language or "TOROT" in self.language:
              if sentence[i]["head"] == 0:
                  sentence[i]["dep"] = "root"

#           if self.splitLemmas:
 #             sentence[i]["lemmas"] = sentence[i]["lemma"].split("+")

  #         if self.storeMorph:
   #           sentence[i]["morph"] = sentence[i]["morph"].split("|")

    #       if self.splitWords:
     #         sentence[i]["words"] = sentence[i]["word"].split("_")


           sentence[i]["dep"] = sentence[i]["dep"].lower()

           result.append(sentence[i])
 #          print sentence[i]
        return (tree,result)



