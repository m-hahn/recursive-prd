import stanfordnlp
#stanfordnlp.download('hi')
nlp = stanfordnlp.Pipeline(processors = "tokenize,lemma,pos", model_path="/u/scr/mhahn/software/stanfordnlp_resources/hi_hdtb_models/", lang="hi", use_gpu=True)

hindi_doc = nlp("""केंद्र की मोदी सरकार ने शुक्रवार को अपना अंतरिम बजट पेश किया. कार्यवाहक वित्त मंत्री पीयूष गोयल ने अपने बजट में किसान, मजदूर, करदाता, महिला वर्ग समेत हर किसी के लिए बंपर ऐलान किए. हालांकि, बजट के बाद भी टैक्स को लेकर काफी कन्फ्यूजन बना रहा. केंद्र सरकार के इस अंतरिम बजट क्या खास रहा और किसको क्या मिला, आसान भाषा में यहां समझें""")

# https://www.analyticsvidhya.com/blog/2019/02/stanfordnlp-nlp-library-python/
PATH = "/u/scr/mhahn/FAIR18/WIKIPEDIA/hindi/"
for partition in ["test", "valid", "train"]:
  lineCounter = 0

  with open(PATH+"/hindi-"+partition+"-tokenized.txt", "w") as outFile:
    with open(PATH+"/hindi-"+partition+".txt", "r") as inFile:
      buff = ""
      for line in inFile:
         lineCounter += 1
         buff = buff+" \n\n "+line.strip()
         if len(buff) > 100000:
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
           print("\t".join([wrd.text, wrd.pos, wrd.lemma]), file=outFile)
   
  
   
