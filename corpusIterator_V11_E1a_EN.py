from paths import WIKIPEDIA_HOME
import random
 

def load(language, partition="train", removeMarkup=True, tokenize=True):
  assert language == "english"
  with open("/u/scr/mhahn/recursive-prd/VSLK_LCP/E1a_EN_SPR/data/e1a_en_spr_data.txt", "r") as inFile:
     text = [x.split(" ") for x in inFile.read().strip().split("\n")]
  header = ["subject","expt","item","condition","position","word","rt"] # according to VSLK_LCP/E1a_EN_SPR/analysis/rcode/processdata.R
  header = dict(zip(header, range(len(header))))
  chunk = []

  chunk_line_numbers = []
  for linenum, line in enumerate(text):
     words = line[header['word']]
     words = words.split("_")
     for word in words:

        if tokenize:
            if word[-1] in [".", ","]:
               chunk.append(word[:-1].lower())
               chunk.append(word[-1].lower())
               chunk_line_numbers.append(linenum)
               chunk_line_numbers.append(linenum)
            else:
               chunk.append(word.lower())
               chunk_line_numbers.append(linenum)
        else:
           for char in " "+word:
               chunk.append(char.lower())
               chunk_line_numbers.append(linenum)

#  print(chunk)
  yield chunk, chunk_line_numbers

def test(language, removeMarkup=True):
  return load(language, "test", removeMarkup=removeMarkup)



if __name__ == '__main__':
    stream = test("english", 1)
    char_vocab_path = {"german" : "vocabularies/german-wiki-word-vocab-50000.txt", "italian" : "vocabularies/italian-wiki-word-vocab-50000.txt", "english" : "vocabularies/english-wiki-word-vocab-50000.txt"}["english"]
    
    with open(char_vocab_path, "r") as inFile:
         itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
    stoi = dict([(itos[i],i) for i in range(len(itos))])
    notIn = 0
    total = 0
    next(stream)

