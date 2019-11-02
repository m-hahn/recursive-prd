import os
import sys

language = sys.argv[1]

path = "/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/"
models = sorted([x for x in os.listdir(path) if x.startswith("estimates-"+language+"_") and "words_NoNewWeightDrop_RunTest.py" in x])

import subprocess

print("results/models_vanillaLSTM_"+language+".tsv")
with open("results/models_vanillaLSTM_"+language+".tsv", "w") as outFile:
 print >> outFile, "\t".join(["Model", "FileName", "AveragePerformance"])
 for model in models:
   model2 = model.split("_")
   ID = model2[-3]
   script = ("_".join(model2[1:-4])).replace("_RunTest", "")
   assert "REAL" == model2[-2]
   with open(path+model, "r") as inFile:
       args = next(inFile)
       surprisal = float(next(inFile).strip())

       print >> outFile, "\t".join([str(x) for x in [ID, script, surprisal]])


