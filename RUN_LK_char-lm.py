# python char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-WHITESPACE.py --language english  --batchSize 128 --char_dropout_prob 0.001 --char_embedding_size 200 --char_noise_prob 0.0 --hidden_dim 1024 --language english --layer_num 3 --learning_rate 3.6  --lr_decay 0.95 --save-to wiki-english-nospaces-bptt-WHITESPACE-MYID --sequence_length 80 --verbose True --weight_dropout_hidden 0.01 --weight_dropout_in 0.0
# wiki-english-nospaces-bptt-WHITESPACE-732605720.pth.tar

# python char-lm-ud-stationary-completion.py --language english --batchSize 128 --char_embedding_size 200 --hidden_dim 1024 --layer_num 3 --weight_dropout_in 0.1 --weight_dropout_hidden 0.2 --char_dropout_prob 0.0 --char_noise_prob 0.01 --learning_rate 0.2 --load-from wiki-english-nospaces-bptt-282506230 

from paths import WIKIPEDIA_HOME
from paths import LOG_HOME
from paths import CHAR_VOCAB_HOME
from paths import MODELS_HOME
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--load-from", dest="load_from", type=str)
parser.add_argument("--section", type=str)


import random

parser.add_argument("--batchSize", type=int, default=1)
parser.add_argument("--char_embedding_size", type=int, default=200)
parser.add_argument("--hidden_dim", type=int, default=1024)
parser.add_argument("--layer_num", type=int, default=2)
parser.add_argument("--weight_dropout_in", type=float, default=0.1)
parser.add_argument("--weight_dropout_hidden", type=float, default=0.2)
parser.add_argument("--char_dropout_prob", type=float, default=0.0)
parser.add_argument("--char_noise_prob", type = float, default=0.01)
parser.add_argument("--learning_rate", type = float, default= 0.2)
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=80)
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([0.5, 0.7, 0.9, 0.95, 0.98, 0.98, 0.98, 0.98, 1.0]))


import math

args=parser.parse_args()


print(args)



import corpusIteratorLK



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

try:
   with open(CHAR_VOCAB_HOME+"/char-vocab-wiki-"+args.language, "r") as inFile:
     itos = inFile.read().strip().split("\n")
except FileNotFoundError:
    assert False

itos.append(" ")
assert " " in itos
print(itos)
stoi = dict([(itos[i],i) for i in range(len(itos))])

itos_complete = ["???", "???", "OOV"] + itos


import random


import torch

print(torch.__version__)

from weight_drop import WeightDrop


rnn = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num).cuda()

rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
print(rnn_parameter_names)
#quit()


rnn_drop = WeightDrop(rnn, [(name, args.weight_dropout_in) for name, _ in rnn.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn.named_parameters() if name.startswith("weight_hh_")])

output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()

char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.char_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)

modules = [rnn, output, char_embeddings]
def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

parameters_cached = [x for x in parameters()]


learning_rate = args.learning_rate

optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) # 0.02, 0.9

named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings, "optim" : optim}

if args.load_from is not None:
  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      module.load_state_dict(checkpoint[name])
else:
  assert False

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout



def prepareDatasetChunks(data, train=True):
      numeric = [0]
      count = 0
      print("Prepare chunks")
      numerified = []
      line_numbers = []
      for chunk, chunk_line_numbers in data:
       for char, linenum in zip(chunk, chunk_line_numbers):
         count += 1
         print(char)
         numerified.append((stoi[char]+3 if char in stoi else 2))
         line_numbers.append(linenum)

       if len(numerified) > (args.batchSize*args.sequence_length):
         sequenceLengthHere = args.sequence_length

         cutoff = int(len(numerified)/(args.batchSize*sequenceLengthHere)) * (args.batchSize*sequenceLengthHere)
         numerifiedCurrent = numerified[:cutoff]
         numerified = numerified[cutoff:]

         line_numbersCurrent = line_numbers[:cutoff]
         line_numbers = line_numbers[cutoff:]
        
         numerifiedCurrent = torch.LongTensor(numerifiedCurrent).view(args.batchSize, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()
         line_numbersCurrent = torch.LongTensor(line_numbersCurrent).view(args.batchSize, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()
         numberOfSequences = numerifiedCurrent.size()[0]
         for i in range(numberOfSequences):
             yield numerifiedCurrent[i], line_numbersCurrent[i]
         hidden = None
       else:
         print("Skipping")


completeData = []



hidden = None

zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize)]).cuda().view(1,args.batchSize)
beginning = None

def forward(numericAndLineNumbers, train=True, printHere=False):
      global hidden
      global beginning
      if hidden is None or (train and random.random() > 0.9):
          hidden = None
          beginning = zeroBeginning
      elif hidden is not None:
          hidden = tuple([Variable(x.data).detach() for x in hidden])

      numeric, lineNumbers = numericAndLineNumbers


      numeric = torch.cat([beginning, numeric], dim=0)

      beginning = numeric[numeric.size()[0]-1].view(1, args.batchSize)


      input_tensor = Variable(numeric[:-1], requires_grad=False)
      target_tensor = Variable(numeric[1:], requires_grad=False)
      

    #  print(char_embeddings)
      #if train and (embedding_full_dropout_prob is not None):
      #   embedded = embedded_dropout(char_embeddings, input_tensor, dropout=embedding_full_dropout_prob, scale=None) #char_embeddings(input_tensor)
      #else:
      embedded = char_embeddings(input_tensor)
      if train:
         embedded = char_dropout(embedded)

      out, hidden = rnn_drop(embedded, hidden)
#      if train:
#          out = dropout(out)

      logits = output(out) 
      log_probs = logsoftmax(logits)
   #   print(logits)
  #    print(log_probs)
 #     print(target_tensor)

      
      lossTensor = print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, args.batchSize)
      losses = lossTensor.data.cpu().numpy()




      for i in range(0,args.sequence_length): #range(1,maxLength+1): # don't include i==0
         j = 0
         numericCPU = numeric.cpu().data.numpy()
         lineNum = int(lineNumbers[i][j])

         print (i, itos_complete[numericCPU[i+1][j]], losses[i][j], lineNum)

         while lineNum >= len(completeData):
             completeData.append([[], 0])
         completeData[lineNum][0].append(itos_complete[numericCPU[i+1][j]])
         completeData[lineNum][1] += losses[i][j]


      return None, target_tensor.view(-1).size()[0]




import time

testLosses = []

if True:
   rnn_drop.train(False)


   test_data = corpusIteratorLK.load(args.language, args.section, tokenize=False)
   print("Got data")
   test_chars = prepareDatasetChunks(test_data, train=False)


     
   test_loss = 0
   test_char_count = 0
   counter = 0
   hidden, beginning = None, None
   while True:
       counter += 1
       try:
          numeric = next(test_chars)
       except StopIteration:
          break
       printHere = (counter % 50 == 0)
       loss, numberOfCharacters = forward(numeric, printHere=printHere, train=False)
       test_char_count += numberOfCharacters
   testLosses.append(test_loss/test_char_count)
   print(testLosses)


with open("output/"+args.section+"_"+args.load_from, "w") as outFile:
   print("\t".join(["LineNumber", "RegionLSTM", "Surprisal"]), file=outFile)
   for num, entry in enumerate(completeData):
     print("\t".join([str(x) for x in [num, "".join(entry[0]), entry[1]]]), file=outFile)



