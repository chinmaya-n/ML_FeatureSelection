#!/bin/bash

# generate models for training files
rankAlgo="pcc"	# s2n - ttest - pcc
norm="true" # true; false

# iterate for each topN
for topN in 1 5 10 20 50 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000
do
	# classify linear egs	
	./svm_classify ./data/${rankAlgo}_rank${topN}_Norm${norm}_svm.valid ./models/${rankAlgo}_rank${topN}_Norm${norm}_svm.model ./results/${rankAlgo}_rank${topN}_Norm${norm}_svm.result
	echo "Wrote file: ./models/${rankAlgo}_rank${topN}_svm.model"
	echo " "
done
