import os

for language in ["english", "german"]:
   path = "/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/"
   models = [x for x in os.listdir(path) if x.startswith("estimates-"+language+"_") and "words_NoNewWeightDrop_RunTest.py" in x]
   
   import subprocess
   
   for model in models:
      model2 = model.split("_")
      ID = model2[-3]
      script = ("_".join(model2[1:-4])).replace("_RunTest", "")
      assert "REAL" == model2[-2]
      #print(ID, script)
      with open(path+model, "r") as inFile:
          args = next(inFile)
          surprisal = float(next(inFile).strip())
   
          for section in {"english" : [], "german" : ["E3_Adapted"]}[language]: # English: "E1", "E1a", "E5", "E6", "E1_EitherVerb", German : "E3"
            command = ["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "RUN_V11_"+script, "--language="+language, "--load-from="+ID, "--section="+section]
            subprocess.call(command)
   
