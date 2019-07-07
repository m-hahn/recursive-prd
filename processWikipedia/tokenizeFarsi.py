import stanfordnlp
#stanfordnlp.download('fa')
nlp = stanfordnlp.Pipeline(processors = "tokenize,lemma,pos", model_path="/u/scr/mhahn/software/stanfordnlp_resources/fa_hdtb_models/", lang="fa", use_gpu=True)


# https://www.analyticsvidhya.com/blog/2019/02/stanfordnlp-nlp-library-python/
PATH = "/u/scr/mhahn/FAIR18/WIKIPEDIA/farsi/"
for partition in ["test", "valid", "train"]:
  lineCounter = 0

  with open(PATH+"/farsi-"+partition+"-tagged.txt", "w") as outFile:
    with open(PATH+"/farsi-"+partition+".txt", "r") as inFile:
      buff = ""
      for line in inFile:
         lineCounter += 1
         buff = buff+" \n\n "+line.strip()
         if len(buff) > 150000:
           print(partition, lineCounter, len(buff))
           doc = nlp(buff)
           buff = ""
           for sent in doc.sentences:
             for wrd in sent.words:
  #              if wrd.text is None or wrd.pos is None or wrd.lemma is None:
   #                  print([wrd.text, wrd.pos, wrd.lemma])
                
                print("\t".join([wrd.text, wrd.pos, wrd.lemma if wrd.lemma is not None else "NONE"]), file=outFile)
   
      doc = nlp(buff)
      buff = ""
      for sent in doc.sentences:
        for wrd in sent.words:
           print("\t".join([wrd.text, wrd.pos, wrd.lemma if wrd.lemma is not None else "NONE"]), file=outFile)
   
  
   
