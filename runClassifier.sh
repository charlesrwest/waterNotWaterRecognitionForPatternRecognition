#! /bin/bash
echo "Starting the SVM with different parameters. This may take a while"
cp "RBFvalues.txt" "RBFvaluesBackup.txt"
cmake ./ && make
for X in `awk -F: '{print $1}' ./RBFvaluesBackup.txt`
do 
	./bin/classifierEvaluator ./data/trainingDataIndex.json ./output/ >> svmOutput.txt
	tail -n +2 "RBFvalues.txt" > "RBFvalues.tmp" && mv "RBFvalues.tmp"  "RBFvalues.txt"
done
