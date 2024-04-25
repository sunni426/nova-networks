#! /bin/bash

# cd hpa_singlecell

python main.py train -i b3 -j jakiro/sin_exp5_b3_rare.yaml
python main.py train -i b5 -j jakiro/sin_exp5_b5_rare.yaml
python main.py train -i r50 -j jakiro/sin_exp5_r50d_rarex2.yaml
python main.py train -i r200d -j jakiro/sin_exp5_r200d_rarex2.yaml
python main.py train -i se50 -j jakiro/sin_exp5_rare-se50.yaml