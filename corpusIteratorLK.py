from paths import WIKIPEDIA_HOME
import random
 

def load(language, section, partition="train", removeMarkup=True, tokenize=True):
  assert language == "german"
  with open("/u/scr/mhahn/CODE/StatSigFilter/data/" + section, "r") as inFile:
     text = [x.split("\t") for x in inFile.read().strip().split("\n")]
  chunk = []
  header = dict(zip(text[0], range(len(text[0]))))
  text = text[1:]
  chunk_line_numbers = []
  for linenum, line in enumerate(text):
     words = line[header['"region"']]
     words = words[1:-1].split("_")
     for word in words:
        if word in ["N", "Y"]:
           continue

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

def test(language, section, removeMarkup=True):
  return load(language, section, "test", removeMarkup=removeMarkup)



if __name__ == '__main__':
    stream = test("german")
    char_vocab_path = {"german" : "vocabularies/german-wiki-word-vocab-50000.txt", "italian" : "vocabularies/italian-wiki-word-vocab-50000.txt", "english" : "vocabularies/english-wiki-word-vocab-50000.txt"}["german"]
    
    with open(char_vocab_path, "r") as inFile:
         itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
    stoi = dict([(itos[i],i) for i in range(len(itos))])
    notIn = 0
    total = 0
    for word in next(stream):
      total += 1
      if word not in stoi:
        print(word)
        notIn += 1.0
    print(notIn / total)





