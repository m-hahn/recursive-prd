from paths import WIKIPEDIA_HOME
import random
 

itos_labels = []
stoi_labels = {}

def load(language, partition="train", removeMarkup=True, tokenize=True):
  assert language == "english"
  with open("/u/scr/mhahn/recursive-prd/BarteketalJEP2011data/gg-spr06-data.txt", "r") as inFile:
     text = [x.split("\t") for x in inFile.read().strip().split("\r")]
  print(text)
  header = ["subj", "expt", "item", "condition", "roi", "word", "RT", "embedding",  "intervention"] # based on master.tex
  header = dict(zip(header, range(len(header))))
  for line in text:
     print(line)
     assert len(line) == len(header)
  chunk = []
  chunk_line_numbers = []
  for linenum, line in enumerate(text):
     word = line[header['word']]
#     words = words[1:-1].split("_")
     if True:
#        if word in ["N", "Y"]:
 #          continue
        assert "'" not in word, "TODO have to deal with this"
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

  print(chunk)
  yield chunk, chunk_line_numbers

def test(language, removeMarkup=True, tokenize=True):
  return load(language, "test", removeMarkup=removeMarkup, tokenize=tokenize)



if __name__ == '__main__':
    stream = test("english")
    char_vocab_path = {"german" : "vocabularies/german-wiki-word-vocab-50000.txt", "italian" : "vocabularies/italian-wiki-word-vocab-50000.txt", "english" : "vocabularies/english-wiki-word-vocab-50000.txt"}["english"]
    
    with open(char_vocab_path, "r") as inFile:
         itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
    stoi = dict([(itos[i],i) for i in range(len(itos))])
    notIn = 0
    total = 0
    chunk, chunk_line_numbers = next(stream)
    for word in chunk:
      total += 1
      print(word, word in stoi)
      if word not in stoi:
        print(word)
        notIn += 1.0
    print(notIn / total)





