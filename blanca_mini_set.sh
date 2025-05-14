#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=24:00:00
#SBATCH --partition=blanca-curc-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=run_code-%j.out
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="asum8093@colorado.edu"

module purge

module load anaconda
module load cuda/12.1.1
#cd /scratch/alpine/asum8093/LiteOntonotesTraining/LiteOntonotesTraining/data/
cd /scratch/alpine/asum8093/LiteOntonotesTraining/LiteOntonotesTraining/

conda activate py38-pt1131-cuda117

echo "== This is the scripting step! =="

#pip install datasets
#cd data/

#ls

#python ./process_ultrafine.py
#cd ../
data_dir="data/small_processed_data"
output_dir="small_output"
device=0

#python3 lite.py --data_dir "data/small_processed_data" \
#                             --output_dir "small_output" \
#                             --train_batch_size 4 \
#                             --num_train_epochs 100 \
#                             --margin 0.1 \
#                             --save_epochs 10 \
#                             --learning_rate 1e-6 \
#                             --lamb 0.05

#mkdir model_dir
#mv /scratch/alpine/asum8093/LiteOntonotesTraining/LiteOntonotesTraining/small_output/16_18_50_Apr_24_2025_batch4_margin0.1_lr1e-06lambda0.05/epochs100/model ./model_dir/model

#python3 eval.py \
#                             --model_dir "./model_dir" \
#                             --eval_data_path "./data/small_processed_data/dev_processed.json" \
#                             --type_vocab_file "./data/small_processed_data/types.txt" \
#                             --batch 4

python3 eval.py \
                             --model_dir "./model_dir/small_model_100_epoch/" \
                             --eval_data_path "./data/bins_processed_data/test_split_q1_processed.json" \
                             --type_vocab_file "./data/bins_processed_data/types.txt" \
                             --batch 4

echo "=============== DDEV 1 ========================="


python3 eval.py \
                             --model_dir "./model_dir/small_model_100_epoch/" \
                             --eval_data_path "./data/bins_processed_data/test_split_q2_processed.json" \
                             --type_vocab_file "./data/bins_processed_data/types.txt" \
                             --batch 4

echo "=============== DDEV 2 ========================="

python3 eval.py \
                             --model_dir "./model_dir/small_model_100_epoch/" \
                             --eval_data_path "./data/bins_processed_data/test_split_q3_processed.json" \
                             --type_vocab_file "./data/bins_processed_data/types.txt" \
                             --batch 4

echo "=============== DDEV 3 ========================="

python3 eval.py \
                             --model_dir "./model_dir/small_model_100_epoch/" \
                             --eval_data_path "./data/bins_processed_data/test_split_q4_processed.json" \
                             --type_vocab_file "./data/bins_processed_data/types.txt" \
                             --batch 4

echo "== End of Eval code =="

python3 result.py --dev "./model_dir/small_model_100_epoch/Evaluation_dev_processed.json" \
                   --test "./model_dir/small_model_100_epoch/Evaluation_test_split_q1_processed.json" \
                   --model_dir "./model_dir/small_model_100_epoch/" \
                   --threshold_step 0.05

echo "=============== TEST 1 ========================="

python3 result.py --dev "./model_dir/small_model_100_epoch/Evaluation_dev_processed.json" \
                   --test "./model_dir/small_model_100_epoch/Evaluation_test_split_q2_processed.json" \
                   --model_dir "./model_dir/small_model_100_epoch/" \
                   --threshold_step 0.05

echo "=============== TEST 2 ========================="

python3 result.py --dev "./model_dir/small_model_100_epoch/Evaluation_dev_processed.json" \
                   --test "./model_dir/small_model_100_epoch/Evaluation_test_split_q3_processed.json" \
                   --model_dir "./model_dir/small_model_100_epoch/" \
                   --threshold_step 0.05

echo "=============== TEST 3 ========================="

python3 result.py --dev "./model_dir/small_model_100_epoch/Evaluation_dev_processed.json" \
                   --test "./model_dir/small_model_100_epoch/Evaluation_test_split_q4_processed.json" \
                   --model_dir "./model_dir/small_model_100_epoch/" \
                   --threshold_step 0.05

echo "=============== TEST 4 ========================="

#wget http://nlp.cs.washington.edu/entity_type/data/ultrafine_acl18.tar.gz

#tar -xvzf ultrafine_acl18.tar.gz

#python run_code.py 

echo "== End of Job =="