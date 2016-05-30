# 1_60 1000 Sample Cascades
# Keep learned data in model
# Number of Batches MF == Number of Batches Preparer

# Example 10 Batches
# W/o normalization
# W/o saving stats

cd ../src/

./mondrianforest_demo.py --dataset Sample_1000_Cascades_1_60 --n_mondrians 50 --budget -1 --normalize_features 0 --save 0 --data_path ../process_data/Sample_1000_Cascades_1_60/LMW/ --n_minibatches 10 > results_stats/LMW_Light_Results.txt

