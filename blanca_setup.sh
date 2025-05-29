#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=24:00:00
#SBATCH --partition=blanca-curc-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=run_code_epoch9-%j.out
#SBATCH --mail-type="ALL"
#SBATCH --mail-user="asum8093@colorado.edu"

module purge

module load anaconda
module load cuda/12.1.1
cd /scratch/alpine/asum8093/LiteOntonotesTraining/LiteOntonotesTraining/

conda activate py38-pt1131-cuda117

echo "== This is the scripting step! =="

#pip install datasets
#wget http://nlp.cs.washington.edu/entity_type/data/ultrafine_acl18.tar.gz
#tar -xvzf ultrafine_acl18.tar.gz
#python run_code.py 
#python process_ultrafine.py
#cd ../

git clone https://github.com/luka-group/lite.git

cd ./lite/data

wget http://nlp.cs.washington.edu/entity_type/data/ultrafine_acl18.tar.gz

tar -xvzf ultrafine_acl18.tar.gz

python3 process_ultrafine.py

cd ../

data_dir="data/processed_data"
output_dir="output"
device=0

# Training command
python3 lite.py --data_dir "data/processed_data" \
                             --output_dir "output" \
                             --train_batch_size 2 \
                             --num_train_epochs 9 \
                             --margin 0.1 \
                             --save_epochs 1 \
                             --learning_rate 1e-6 \
                             --lamb 0.05 \
                             --resume_from_checkpoint model_dir/big_model_8_epoch/model \
                             --resume_epoch 8
                             --resume_step 0


# Evaluation command---------------------
#python3 eval.py \
#                             --model_dir "./model_dir" \
#                             --eval_data_path "./data/processed_data/dev_processed.json" \
#                             --type_vocab_file "./data/processed_data/types.txt" \
#                             --batch 4

#python3 eval.py \
#                             --model_dir "./model_dir" \
#                             --eval_data_path "./data/processed_data/test_processed.json" \
#                             --type_vocab_file "./data/processed_data/types.txt" \
#                             --batch 4


# Test command --------------------------
#python3 result.py --dev "./model_dir/Evaluation_dev_processed.json" \
#                   --test "./model_dir/Evaluation_test_processed.json" \
#                   --model_dir "./model_dir/" \
#                   --threshold_step 0.05



echo "== End of Job =="