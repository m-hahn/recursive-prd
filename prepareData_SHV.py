# PROBLEM for Farsi: the zero-width no-joiner isn't reflected in the data. 
# Also, there are some erroneous word boundaries in the PDF encoding, which reflect as erroneous tokenization here, and partly as OOVs (e.g., Item 7c, Naghdi has nagh and di separated).
# Could try this (this one has a paper, so should be mature): https://github.com/ICTRC/Parsivar

# An example is: ﻦﻣی<200c>ﺕﻭﺎﻨﺴﺘﻣ, which is in vocabulary, but is erroneously separated


# Also another problem: The initil character (sometimes?) disappears. E.g. لهام is the name Elham, with the first character (Alif) chopped off. Occurs in Items 11 a/b.

# Segments the data using the same tokenizer as used when training the language model
# Checks for OOVs

import stanfordnlp
#stanfordnlp.download('hi')
nlp = stanfordnlp.Pipeline(processors = "tokenize,lemma,pos", model_path="/u/scr/mhahn/software/stanfordnlp_resources/fa_hdtb_models/", lang="fa", use_gpu=True)

from parsivar import Normalizer
my_normalizer = Normalizer(statistical_space_correction=True)

language = "farsi"

char_vocab_path = "/u/scr/mhahn/FAIR18/WIKIPEDIA/farsi/farsi-wiki-word-vocab.txt"
#char_vocab_path = "vocabularies/"+language.lower()+"-wiki-word-vocab-50000.txt"

print("Loading vocab")

with open(char_vocab_path, "r") as inFile:
     itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")]
stoi = dict([(itos[i],i) for i in range(len(itos))])

print("Done loading vocab")

with open("vocabularies/char-vocab-wiki-"+language, "r") as inFile:
     itos_chars = [x for x in inFile.read().strip().split("\n")]
stoi_chars = dict([(itos_chars[i],i) for i in range(len(itos_chars))])



with open("stimuli/safavi_etal_2016_persian/items_expt1.txt", "r") as inFile:
 with open("stimuli/safavi_etal_2016_persian/items_expt1_tokenized.txt", "w") as outFile:
   print("\t".join([str(z) for z in ["ItemCondition", "OriginalTokenNumber", "Region", "TokenizedTokenNumber", "TokenizedWord", "Experiment", "Item", "Condition"]]), file=outFile)

   typ = "1 1 1"
   for line in inFile:
      if len(line) > 3:
         text = line.strip()
#         print(text)
         text = [x for x in text.replace("\u202b", "").replace("\u202c", "").split(" ")][::-1]
         words = [x.split("_") for x in text]
         text = []
         for w in words:
           for v in w:
            text.append(v)
 #        print(text)

         i=0
         while i+1 < len(text):
             if text[i]+'\u200c'+text[i+1] in stoi:
#                  print("@@@@@@@@@@@@@@@@@@@@@@@")
                  rank1=stoi.get(text[i], -1)
                  rank2=stoi.get(text[i+1], -1)
                  rank3=stoi.get(text[i]+'\u200c'+text[i+1])
#                  print(stoi.get(text[i], -1), stoi.get(text[i+1], -1), stoi.get(text[i]+'\u200c'+text[i+1]), text[i]+'\u200c'+text[i+1])
                  if (rank1 == -1 or rank2 == -1) or rank3 < rank1 or rank3 < rank2:
                      print("REPLACING")
                      text[i] = text[i]+'\u200c'+text[i+1]
                      del text[i+1]
                      i-=1
             i+=1

         textstr = " ".join(text)
         # RUN a tetx normalizer for correcting half-space
         normalized = textstr #my_normalizer.normalize(textstr)
  #       print(textstr)
   #      print(normalized)
         textstr = normalized #assert textstr == normalized
         doc = nlp(textstr[1:])

         text_tokenized = []
         for sent in doc.sentences:
           for wrd in sent.words:
              text_tokenized.append(wrd.text)
              if wrd.text not in stoi:
                  print("\t\t".join([str(x) for x in [wrd.text, wrd.lemma, wrd.pos, "OOV========================"]]))



#         print(text_tokenized)
 #        print(len(text), len(text_tokenized))
#         assert len(text) <= len(text_tokenized)
         posTxt = 0
         posWrd = 1
         i = 0


#         for y in words:
##             print(y, posWrd, posTxt)
#             
#             for _ in y:
#                if posWrd == len(text_tokenized[posTxt]):
#                    #print("\t".join([str(z) for z in [typ, i, posTxt, y, text_tokenized[posTxt]]])) #, file=outFile)
#                    if text_tokenized[posTxt] not in stoi:
#                        print("OOV", typ, text_tokenized[posTxt])
#                        print("==========================")
#                        print("==========================")
#
#                    posTxt += 1
#                    posWrd = 1
#                else:
#                    posWrd += 1
#
#             i += 1


