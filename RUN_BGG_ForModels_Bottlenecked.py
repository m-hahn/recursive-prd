import os
from math import log 

for language in ["english"]:
   path = "/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/"
   models = [x for x in os.listdir(path) if x.startswith("estimates-"+language+"_") and "LogExp11_CHOSEN" in x]
   
   import subprocess
   with open("results/models_bottlenecked_"+language, "w") as outFile:
    print("\t".join(["ID", "Model", "Script", "Surprisal", "LogBeta"]), file=outFile)
    for model in models:
      model2 = model.split("_")
#      print(model2)
      ID = model2[-2]
      script = ("_".join(model2[1:-3])).replace("_RunTest", "")
#     print(script)
      assert "REAL.txt" == model2[-1]
      #print(ID, script)
 #     print(ID)
 #     continue 
      with open(path+model, "r") as inFile:
          args = next(inFile)
          if "log_beta" in args:
             log_beta = args[args.index("log_beta"):]
             log_beta = log_beta[log_beta.index("=")+1:log_beta.index(",")]
          else:
             beta = args[args.index("log_beta"):]
             beta = beta[beta.index("=")+1:beta.index(",")]
             beta = str(-log(float(beta)))
          surprisal = float(next(inFile).strip().split(" ")[-1])
          print("\t".join([str(x) for x in [ID, model, script, surprisal, log_beta]]), file=outFile)
 #         continue 
          command = ["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "RUN_BGG_"+script, "--language="+language, "--load_from="+ID]
          print(" ".join(command))
          subprocess.call(command)
   
