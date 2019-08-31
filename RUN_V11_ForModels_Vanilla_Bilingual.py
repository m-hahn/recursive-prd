import os
from math import log 

for language in ["english", "german"]:
   path = "/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/"
   models = [x for x in os.listdir(path) if x.startswith("estimates-"+"englishANDgerman"+"_") and "words_NoNewWeightDrop" in x]
   
   import subprocess
   with open("results/models_vanilla_bilingual", "w") as outFile:
    print("\t".join(["ID", "Model", "Script", "Surprisal"]), file=outFile)
    for model in models:
      model2 = model.split("_")
      ID = model2[-3]
      print(model2)
      print(int(ID))
      script = ("_".join(model2[1:-4])).replace("_RunTest", "")
      assert "REAL" == model2[-2], model2
      with open(path+model, "r") as inFile:
          args = next(inFile)
          surprisal = float(next(inFile).strip().split(" ")[-1])
          print("\t".join([str(x) for x in [ID, model, script, surprisal]]), file=outFile)
 
          for section in {"english" : ["E1", "E1_ColorlessGreen"], "german" : ["E3_Adapted", "E3_ColorlessGreen"]}[language]: # English: "E1", "E1a", "E5", "E6", "E1_EitherVerb", German "E3", 
            command = ["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "RUN_V11_"+script, "--language1=english", "--language2=german", "--chosen_language="+language, "--load_from="+ID, "--section="+section]
            subprocess.call(command)
   
