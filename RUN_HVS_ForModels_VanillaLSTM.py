import os

language = "hindi"

path = "/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/"
models = [x for x in os.listdir(path) if x.startswith("estimates-"+language+"_") and "words_NoNewWeightDrop_RunTest.py" in x]

import subprocess

with open("results/models_vanillaLSTM_"+language+".tsv", "w") as outFile:
 for model in models:
   model2 = model.split("_")
   ID = model2[-3]
   script = ("_".join(model2[1:-4])).replace("_RunTest", "")
   assert "REAL" == model2[-2]
   #print(ID, script)
   with open(path+model, "r") as inFile:
       args = next(inFile)
       surprisal = float(next(inFile).strip())

       print >> outFile, "\t".join([str(x) for x in [ID, script, surprisal]])

       hyperparams = ["--"+x for x in args[10:-2].split(", ") if "myID" not in x and "language" not in x and "batchSize" not in x and "load_from" not in x]

       command = ["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "RUN_HVS_"+script, "--language=hindi", "--load-from="+ID] + hyperparams
       subprocess.call(command)

