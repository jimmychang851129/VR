#!/bin/bash

python gen_tri.py > tmp

cat tmp | grep -v "\-1" > tri.txt

rm -f tmp
