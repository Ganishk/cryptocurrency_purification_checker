#Crypto Currency Transaction Purification Checker

use `./main.py --help` to get help

syntax: `main.py [-h] -d DATASET [-p PREPROCESS] [-m MODEL] [-r RESULT] [2> LOG_WARNINGS]`

  -h, --help            show this help message and exit\
  -d, --dataset DATASET\
  -p, --preprocess PREPROCESS\
  -m, --model MODEL\
  -r, --result RESULT\
  2> is used to log the warning messages, use /dev/null or nul or other dev null to supress it.

\*preprocessed.obj file once created can be reused again to train the data directly instead of
preprocessing it again.

\*mode.obj file once created can be used to assess all the previous created models, and can be
directly used for testing

## Citations
- BitcoinHeistRansomwareAddressDataset. (2020). UCI Machine Learning Repository.
