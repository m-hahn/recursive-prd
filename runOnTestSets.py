
# NOTE this doesn't take care of hyperparameters. For regularization parameters this doesn't matter, as they are turned off during evaluation.

LANGUAGE = "russian"

scripts = []
#scripts.append("char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py")
#scripts.append("char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_Search.py")
#scripts.append("char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_Farsi.py")
scripts.append("char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_BPE.py")

#scripts.append("char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_RunTest.py")


import os

BASE_DIR = "/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/"

estimates = [x for x in os.listdir(BASE_DIR) if x.startswith("estimates-") and "RunTest" not in x]
import subprocess
for name in estimates:
   n = name[10:].replace("REAL_REAL", "REAL-REAL").split("_")
   language = n[0]
   if language != LANGUAGE:
       continue
   ID = n[-2]
   script = "_".join(n[1:-3])
   #print(n)
   #print(ID)
   print(script)
   if script in scripts:
       script = script.replace("_Search.py",".py")
       script = script.replace("_Farsi.py",".py")

       with open(BASE_DIR+name, "r") as inFile:
#           next(inFile)
           hyperparams = ["--"+x for x in next(inFile)[10:-2].split(", ") if "myID" not in x and "language" not in x]
 #          print(hyperparams)
#       continue

       command = ["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", script.replace(".py", "_RunTest.py"), "--language", language, "--load-from", ID] + hyperparams
       print(command)
#       continue
       subprocess.call(command)

