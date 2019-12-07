import sys
import os
import subprocess

FILE = "autoencoder2_mlp_bidir_Erasure_SelectiveLoss.py"

PATH = "/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/"
files = [x for x in os.listdir(PATH) if x.startswith("estimates-"+"english"+"_"+FILE+"_")]

for name in files:
   args = next(open(PATH+name, "r")).strip().replace("'", "")
   args = [x for x in args[10:-1].split(", ") if not (x.startswith("load_from"))]
   id_ = [x for x in args if x.startswith("myID=")][0].split("=")[1]
   args.append("load-from="+id_)
   subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "TEST_"+FILE] + ["--"+x for x in args])
