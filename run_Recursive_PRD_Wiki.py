import subprocess

import sys

import random
import math

while True:
   language = sys.argv[1]
   languageCode = language
   dropout_rate = random.choice([0.0,0.0,0.0,0.1,0.2])
   emb_dim = random.choice([50,100,200,300])
   rnn_dim = random.choice([64,128,256])
   rnn_layers = random.choice([1])
   lr_lm = random.choice([0.0001, 0.0001, 0.0001, 0.0001, 0.0002, 0.0005, 0.001, 0.001,  0.001])
   model = random.choice(["REAL", "REVERSE", "RANDOM_BY_TYPE", "RANDOM_BY_TYPE", "RANDOM_BY_TYPE"])
   input_dropoutRate = random.choice([0.0,0.0,0.0,0.1,0.2])

   batchSize = random.choice([16, 32,64])
   horizon = 10
   beta = math.exp(random.uniform(-7, -1.5))
   print(beta)
   flow_length = random.choice([1,1,1,2])
   flowtype = random.choice(["dsf", "dsf", "ddsf"])
   flow_hid_dim = random.choice([64,128,256, 512])
   command = map(str,["./python27", "yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_POSOnly_Variational_Bottleneck_TwoRNNs_NeuralFlow_Optimizer_DIMENSIONS_SEPARATE_RECURSIVE"+random.choice(["", "", "_DOUBLE"])+".py", language, languageCode, dropout_rate, emb_dim, rnn_dim, rnn_layers, lr_lm, model, input_dropoutRate, batchSize, horizon, beta, flow_length, flowtype, flow_hid_dim])
   print(" ".join(command))
   subprocess.call(command)
  
   
    
   
