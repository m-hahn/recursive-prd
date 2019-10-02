import os

language = "english"

path = "/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/"
models = [x for x in os.listdir(path) if x.startswith("estimates-"+language+"_") and "words_NoNewWeightDrop_RunTest.py" in x]

import subprocess

failedScripts = set()

files = set(os.listdir("."))
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

       print("\t".join([str(x) for x in [ID, script, surprisal]]), file=outFile)

       scriptname = "RUN_Traxler2002_"+script
       if scriptname not in files:
          failedScripts.add(scriptname)
       for section in ["expt1", "expt2", "expt3"]:
         command = ["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", scriptname, "--language=english", "--load-from="+ID, "--section="+section]
         subprocess.call(command)

print(failedScripts)
