
scripts = []
scripts.append("char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py")
#scripts.append("char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_RunTest.py")


import os

BASE_DIR = "/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/"

estimates = [x for x in os.listdir(BASE_DIR) if x.startswith("estimates-") and "RunTest" not in x]
import subprocess
for name in estimates:
   n = name[10:].replace("REAL_REAL", "REAL-REAL").split("_")
   language = n[0]
   ID = n[-2]
   script = "_".join(n[1:-3])
   #print(n)
   #print(ID)
   print(script)
   if script in scripts:
       command = ["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", script.replace(".py", "_RunTest.py"), "--language", language, "--load-from", ID]
       subprocess.call(command)

