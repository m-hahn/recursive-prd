# Segments the data using the same tokenizer as used when training the language model
# Checks for OOVs

import stanfordnlp
#stanfordnlp.download('hi')
nlp = stanfordnlp.Pipeline(processors = "tokenize,lemma,pos", model_path="/u/scr/mhahn/software/stanfordnlp_resources/hi_hdtb_models/", lang="hi", use_gpu=True)



language = "hindi"

char_vocab_path = "vocabularies/"+language.lower()+"-wiki-word-vocab-50000.txt"
with open(char_vocab_path, "r") as inFile:
     itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
stoi = dict([(itos[i],i) for i in range(len(itos))])

with open("vocabularies/char-vocab-wiki-"+language, "r") as inFile:
     itos_chars = [x for x in inFile.read().strip().split("\n")]
stoi_chars = dict([(itos_chars[i],i) for i in range(len(itos_chars))])




with open("stimuli/husain_etal_2014_hindi/final_items_all.txt", "r") as inFile:
 with open("stimuli/husain_etal_2014_hindi/final_items_all_tokenized.txt", "w") as outFile:
   print("\t".join([str(z) for z in ["ItemCondition", "OriginalTokenNumber", "Region", "TokenizedTokenNumber", "TokenizedWord"]]), file=outFile)

   typ = None
   for line in inFile:
      if line.startswith("#"):
          typ = line.strip()
      elif line.startswith("?"):
          continue
      elif len(line) > 2:
#         print(typ ,line.strip())
         text = line.strip()
         text = [x.split("@") for x in text.split(" ")]
         words = [x[0].split("_") for x in text]
         regions = [x[1] if len(x) > 1 else "NONE" for x in text]
         words = zip(words, regions)
#         print(words)
         textstr = ""
         for w in text:
             for x in w[0].split("_"):
                textstr += " "+x
         doc = nlp(textstr[1:])

         text_tokenized = []
         for sent in doc.sentences:
           for wrd in sent.words:
#              print("\t".join([wrd.text, wrd.pos, wrd.lemma if wrd.lemma is not None else "NONE"]))
              text_tokenized.append(wrd.text)
 #        print(textstr)
         #print("====================")
         #print("   ".join([x[0] for x in text]))
         #print("   ".join(text_tokenized))
         posTxt = 0
         posWrd = 1
         i = 0
         for x, r in words:
               for y in x:
              #   if y not in stoi:
                    #print("\t".join((typ, y, str(y in stoi), r))) #([[(y,y in stoi) for y in x] for x in words])
                  #  print("\t".join(("WORD", y, str(len([_ for _ in y])), str(posTxt), str(posWrd))))
                    for _ in y:
                       if posWrd == len(text_tokenized[posTxt]):
                           print("\t".join([str(z) for z in [typ, i, r, posTxt, text_tokenized[posTxt]]]), file=outFile)
#                           print(i, posTxt, posWrd, y, "\t", text_tokenized[posTxt], "XXXXXXXXXXXXXX" if y != text_tokenized[posTxt] else "")
                           if r != "NONE":
                             if ("RC1" in typ and r == "RCVerb") or ("RC2" in typ and r in ["NP1", "RCPn"]) or ("PP1" in typ and r == "HeadNP") or("CP1" in typ and r == "CPLightVerb"):
                                if text_tokenized[posTxt] not in stoi:
                                    print("OOV", r, typ, text_tokenized[posTxt])
                           posTxt += 1
                           posWrd = 1
                       else:
                           posWrd += 1

               i += 1

