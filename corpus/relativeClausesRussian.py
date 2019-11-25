# Fits grammars to orderings found in corpora.
# Michael Hahn, 2019

import random
import sys

import torch.nn as nn
import torch
from torch.autograd import Variable
import math



import os

language = "Russian"

myID = random.randint(0,10000000)

posUni = set()
posFine = set()



from math import log, exp
from random import random, shuffle

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

import os
sys.path.append("..")
from corpusIterator import CorpusIterator

originalDistanceWeights = {}


def makeCoarse(x):
   if ":" in x:
      return x[:x.index(":")]
   return x

def initializeOrderTable():
   orderTable = {}
   keys = set()
   vocab = {}
   distanceSum = {}
   distanceCounts = {}
   depsVocab = set()
   for partition in ["train", "dev"]:
     for sentence in CorpusIterator(language,partition).iterator():
      for line in sentence:
          vocab[line["word"]] = vocab.get(line["word"], 0) + 1
          line["coarse_dep"] = makeCoarse(line["dep"])
          depsVocab.add(line["coarse_dep"])
          posFine.add(line["posFine"])
          posUni.add(line["posUni"])
  
          if line["coarse_dep"] == "root":
             continue
          posHere = line["posUni"]
          posHead = sentence[line["head"]-1]["posUni"]
          dep = line["coarse_dep"]
          direction = "HD" if line["head"] < line["index"] else "DH"
          key = dep
          keyWithDir = (dep, direction)
          orderTable[keyWithDir] = orderTable.get(keyWithDir, 0) + 1
          keys.add(key)
          distanceCounts[key] = distanceCounts.get(key,0.0) + 1.0
          distanceSum[key] = distanceSum.get(key,0.0) + abs(line["index"] - line["head"])
   #print orderTable
   dhLogits = {}
   for key in keys:
      hd = orderTable.get((key, "HD"), 0) + 1.0
      dh = orderTable.get((key, "DH"), 0) + 1.0
      dhLogit = log(dh) - log(hd)
      dhLogits[key] = dhLogit
      originalDistanceWeights[key] = (distanceSum[key] / distanceCounts[key])
   return dhLogits, vocab, keys, depsVocab

import torch.nn as nn
import torch
from torch.autograd import Variable


def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
   line = sentence[position-1]
   # Invariant: these are the gradients relevant at everything starting at the left end of the domain of the current element
   allGradients = gradients_from_the_left_sum

   largestDepth = 0
   # there are the gradients of its children
   if "children_DH" in line:
      for child in line["children_DH"]:
         d2 = recursivelyLinearize(sentence, child, result, allGradients)
         largestDepth = max(d2, largestDepth)
   result.append(line)
   if "children_HD" in line:
      for child in line["children_HD"]:
         d2 = recursivelyLinearize(sentence, child, result, allGradients)
         largestDepth = max(d2, largestDepth)
   return largestDepth + 1

import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()



def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       childrenLinearized = []
       while len(remainingChildren) > 0:
           logits = [0 for x in remainingChildren]
           selected = 0
           childrenLinearized.append(remainingChildren[selected])
           del remainingChildren[selected]
       return childrenLinearized           


def computeConstituentLength(sentence, index):
   length = 1
   for child in sentence[index-1]["children"]:
      length += computeConstituentLength(sentence, child)
   return length
postVerbalObjects = []
preVerbalObjects = []
postVerbalSubjects = []
preVerbalSubjects = []

def orderSentence(sentence, printThings=False):
   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0
   for line in sentence:
      line["coarse_dep"] = makeCoarse(line["dep"])
      if line["coarse_dep"] == "root":
          root = line["index"]
          continue
      if line["coarse_dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         continue
      key = line["coarse_dep"]
      line["dependency_key"] = key
      dhSampled = True
   
      direction = "DH" if dhSampled else "HD"
      if printThings: 
         print("\t".join(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, (str(probability.data.numpy())+"      ")[:8], str(1/(1+exp(-dhLogits[key])))[:8]     ]  )))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])


      
   for line in sentence:
      line["children"] = line.get("children_DH", []) + line.get("children_HD", [])
      arity.append(len(line.get("children_DH", [])) + len(line.get("children_HD", [])))
      if "children_DH" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_DH"][:], False)
         line["children_DH"] = childrenLinearized
      if "children_HD" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_HD"][:], True)
         line["children_HD"] = childrenLinearized
   for line in sentence:
     if line["dep"] == "acl:relcl":
 #       print(line)
        # find subject, object
        assert line["index"] not in line["children"]
        dependents = [sentence[x-1]["dep"] for x in line["children"]]
#        print(dependents)
        morph_dependents = [sentence[x-1]["morph"] for x in line["children"]]
        position_dependents = [x < line["index"] for x in line["children"]]
        words_dependents = [sentence[x-1]["word"] for x in line["children"]]
        lemmas_dependents = [sentence[x-1]["lemma"] for x in line["children"]]
        pos_dependents = [sentence[x-1]["posUni"] for x in line["children"]]
        dependentsTogether = list(zip(line["children"], dependents, words_dependents,lemmas_dependents, pos_dependents, morph_dependents, position_dependents))
        if lemmas_dependents[0] in ["который", 'что']:
           extractedType = dependents[0]
  #         print(extractedType, line["dep"], dependentsTogether )
           subjects = [x for x in dependentsTogether if x[1] == "nsubj" and x[4] == "NOUN"]
           objects = [x for x in dependentsTogether if x[1] == "obj" and x[4] == "NOUN"]
 #          print(subjects)
#           print(objects)
           # length of preverbal objects in subject-extracted RCs
           if len(objects) > 0 and extractedType == "nsubj":
              fullLength = computeConstituentLength(sentence, objects[0][0])
              if objects[0][-1]:
                preVerbalObjects.append(fullLength)
                print(objects[0][4])
              else:
                postVerbalObjects.append(fullLength)
                
           #   print(len(preVerbalObjects), len(postVerbalObjects))

           if len(subjects) > 0 and extractedType == "obj":
              fullLength = computeConstituentLength(sentence, subjects[0][0])
              if subjects[0][-1]:
                preVerbalSubjects.append(fullLength)
              else:
                postVerbalSubjects.append(fullLength)
                print(subjects[0][4])

           print(len(preVerbalSubjects), len(postVerbalSubjects), len(preVerbalObjects), len(postVerbalObjects))


        # subject-extracted
        # object-extracted
   linearized = []
   tree_depth.append(recursivelyLinearize(sentence, root, linearized, Variable(torch.FloatTensor([0.0]))))
   if printThings or len(linearized) == 0:
     print(" ".join(map(lambda x:x["word"], sentence)))
     print(" ".join(map(lambda x:x["word"], linearized)))


   # store new dependency links
   moved = [None] * len(sentence)
   for i, x in enumerate(linearized):
      moved[x["index"]-1] = i
   for i,x in enumerate(linearized):
      if x["head"] == 0: # root
         x["reordered_head"] = 0
      else:
         x["reordered_head"] = 1+moved[x["head"]-1]
   return linearized


dhLogits, vocab, vocab_deps, depsVocab = initializeOrderTable()

posUni = list(posUni)
itos_pos_uni = posUni
stoi_pos_uni = dict(zip(posUni, range(len(posUni))))

posFine = list(posFine)
itos_pos_ptb = posFine
stoi_pos_ptb = dict(zip(posFine, range(len(posFine))))



itos_pure_deps = sorted(list(depsVocab)) 
stoi_pure_deps = dict(zip(itos_pure_deps, range(len(itos_pure_deps))))
   

itos_deps = sorted(vocab_deps)
stoi_deps = dict(zip(itos_deps, range(len(itos_deps))))






words = list(vocab.items())
#print(words)

totalCount = sum(x[1] for x in words)
probs = [float(x[1])/totalCount for x in words]
unigram_entropy = -sum([x*log(x) for x in probs])
#print(unigram_entropy)

sentenceLengths = []

tree_depth = []
arity = []

numberOfSentences = 0

for sentence in CorpusIterator(language,"train").iterator():
   orderSentence(sentence)
   sentenceLengths.append(len(sentence))
#   print(numberOfSentences)
   numberOfSentences += 1
#print(sentenceLengths)
#print(arity)
#print(tree_depth)

def median(x):
   return sorted(x)[int(len(x)/2)]
def mean(x):
   return float(sum(x))/len(x)

print("==================")
print(median(preVerbalSubjects), median(postVerbalSubjects), median(preVerbalObjects), median(postVerbalObjects))
print(mean(preVerbalSubjects), mean(postVerbalSubjects), mean(preVerbalObjects), mean(postVerbalObjects))

