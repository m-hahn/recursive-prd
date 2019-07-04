from paths import WIKIPEDIA_HOME
import random
 

itos_labels = []
stoi_labels = {}

def load(language, partition="train", removeMarkup=True, tokenize=True):
  assert language == "english"

  stimuli = []
  if True:
     with open("stimuli/cunnings-sturt-2018.txt", "r") as inFile:
        text = [x.split("\t") for x in inFile.read().strip().split("\n")]
     stimuli = []
   
     lastLabel = None
     for line in text:
        print(line)
        sentence = line[1]
        sentence = sentence.lower() + " "
        firstSlash = sentence.index("/")
        secondSlash = sentence.find("/", firstSlash+1)
        firstStart = sentence.rfind(" ", 0, firstSlash)
        firstEnd = sentence.find(" ", firstSlash)
   
        secondStart = sentence.rfind(" ", 0, secondSlash)
        secondEnd = sentence.find(" ", secondSlash)
   
        adverb = sentence.find("ly", secondEnd)
        verbStart = adverb+3
        verbEnd = sentence.find(" ", verbStart)
        verb = sentence[verbStart:verbEnd]
        assert sentence.count(verb) == 1
   
   
        part1 = sentence[:firstStart]
        part2 = sentence[firstEnd:secondStart]
        part3 = sentence[secondEnd:]
        w11 = sentence[firstStart:firstSlash]
        w12 = " "+sentence[firstSlash+1:firstEnd]
   
        w21 = sentence[secondStart:secondSlash]
        w22 = " "+sentence[secondSlash+1:secondEnd]
   
   
        version11 = part1 + w11 + part2 + w21 + part3
        version12 = part1 + w11 + part2 + w22 + part3
        version21 = part1 + w12 + part2 + w21 + part3
        version22 = part1 + w12 + part2 + w22 + part3
   
        assert "/" not in version11
        assert "/" not in version12
        assert "/" not in version21
        assert "/" not in version22
   
        print(firstSlash, secondSlash)
        print(version11)
        print(version12)
        print(version21)
        print(version22)
   
        for condition, sent in zip(["11","12","21","22"], [version11, version12, version21, version22]):
           if tokenize:
            sent = sent.replace(".", " .")
            sent = sent.replace(",", " ,")
            sent = sent.strip().split(" ")
           else:
             verbPositionStart = sent.index(verb)
             verbPositionEnd = sent.find(" ", verbPositionStart)
            
           stimuli.append(([],[]))
           for position, word in enumerate(sent):

              if tokenize and word == verb:
                 critical = True
              elif tokenize:
                 critical = False
              elif not tokenize:
                 critical = (position >= verbPositionStart and position <= verbPositionEnd)
              label = (line[0], ["plau_sent", "impl_sent"][int(condition[0])-1], ["plau_dist", "impl_dist"][int(condition[0])-1], "critical_verb" if critical else "other")
      
              for x in label:
                 if x not in stoi_labels:
                   stoi_labels[x] = len(itos_labels)
                   itos_labels.append(x)
              lastLabel = [stoi_labels[x] for x in label]
              stimuli[-1][0].append(word)
              stimuli[-1][1].append(lastLabel)

  chunk, chunk_line_numbers = [], []

  for iteration in range(10):
     random.shuffle(stimuli)
     for ch, lab in stimuli:
       assert len(ch) == len(lab)
       for i in range(len(ch)):
          chunk.append(ch[i])
          chunk_line_numbers.append(lab[i] + [iteration])

  # labels: item, plausibility, distrractor plausibility, region (critical or other), iteration
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
    print(stream)
    print(next(stream))
    for word in stream:
      total += 1
      if word not in stoi:
        print(word)
        notIn += 1.0
    print(notIn / total)





