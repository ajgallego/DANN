#!/bin/bash

gpu=0

TYPE=cnn	# cnn, dann

model=1         # 1, 2, 3, 4, 21, 22
db=mnist        # mnist  signs
select=None     # none  |||  mnist, mnist_m, svhn, syn_numbers, usps   |||   gtsrb, syn_signs
from_db=none
to_db=none
norm=255        # 255 mean standard
e=200
b=128   	# 64 128 256
#lr1=1.0
options=""  # --tsne --truncate --v  -size 40


python -u dann.py -type ${TYPE} \
		-model ${model} -db ${db} -select ${select} \
		-from ${from_db} -to ${to_db} \
		-norm ${norm} \
		-e ${e} -b ${b} \
		-gpu $gpu \
		${options} \
		> out_${TYPE}_model_${model}_${db}_select_${select}_norm_${norm}_e${e}_b${b}_${options}.txt


