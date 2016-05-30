# 1_60 1000 Sample Cascades
# Simulate discarding old data
# Total Number of Runs := (1/SlideSize * (1 − WindowSize)) + 1

# Example for Window 0.2, Slide 0.1
# Note that the statistics are appended to the same file for each window.
# Therefore not only the results, but also the other stats are written down every time.
# Nevertheless, they are still written down in the correct order of execution. 

i=1

cd ../src/

while [ $i -le 9 ]
do
  ./mondrianforest_demo.py --dataset $i --n_mondrians 50 --budget -1 --normalize_features 1 --save 1 --data_path ../process_data/Sample_1000_Cascades_1_60/SW/ --n_minibatches 1 >> results_stats/SW_Results.txt
  i=`expr $i + 1`
done
