import os
import sys
if len(sys.argv) > 1 and sys.argv[1] == "cpu":
  CPU = True
else:
  CPU = False
language = "english"

path = "/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/"
models = [x for x in os.listdir(path) if x.startswith("estimates-"+language+"_") and "words_NoNewWeightDrop_RunTest.py" in x]

import subprocess

with open("results/models_vanillaLSTM_"+language+".tsv", "w") as outFile:
 for model in models:
   model2 = model.split("_")
   ID = model2[-3]
   script = ("_".join(model2[1:-4])).replace("_RunTest", "")
   if CPU:
     script = script.replace(".py", "_CPU.py")
   assert "REAL" == model2[-2]
   #print(ID, script)
   with open(path+model, "r") as inFile:
       args = next(inFile)
       surprisal = float(next(inFile).strip())

       print >> outFile, "\t".join([str(x) for x in [ID, script, surprisal]])

       for section in ["_default"]:
          command = ["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "RUN_SCRC_"+script, "--language=english", "--load-from="+ID, "--section="+section]
          subprocess.call(command)

