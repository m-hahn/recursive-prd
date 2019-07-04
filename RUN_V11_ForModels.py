import os

language = "english"

path = "/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/"
models = [x for x in os.listdir(path) if x.startswith("estimates-"+language+"_")]

import subprocess

with open("results/models"+language+".tsv", "w") as outFile:
 for model in models:
   model2 = model.split("_")
   ID = model2[-2]
   script = "_".join(model2[1:-3])
   with open(path+model, "r") as inFile:
       args = next(inFile)
       surprisal = next(inFile).strip().split(" ")
       if len(surprisal) < 10:
         continue
       next(inFile)

       memory = next(inFile).strip().split(" ")
 #      print(ID, script)
#       print(len(surprisal), surprisal[-1], memory[-1])
       args = args[10:-2].replace("'", "").replace("myID", "load_from").replace(", ", ", --").split(", ")
       beta = [x for x in args if x.startswith("--beta=")]
       if len(beta) > 0:
          beta=beta[0].split("=")[1]
       else:
          beta = "0"
       flow_length = [x for x in args if x.startswith("--flow_length=")]
       if len(flow_length) > 0:
          flow_length = flow_length[0].split("=")[1]
       else:
          flow_length = "0"

       print >> outFile, "\t".join([str(x) for x in [ID, script, len(surprisal), surprisal[-1], memory[-1], beta]])

       args = [x for x in args if "batchSize" not in x]
       print(args)

       command = ["./python27", "RUN_V11_"+script] + args
       subprocess.call(command)

