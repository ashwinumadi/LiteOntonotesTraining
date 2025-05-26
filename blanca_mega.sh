#!/bin/bash
#SBATCH --partition=blanca-blast-lecs
#SBATCH --account=blanca-blast-lecs
#SBATCH --qos=blanca-blast-lecs
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=4000m
#SBATCH --gres=gpu:h100_7g.80gb
#SBATCH --time=5-23:59:59
#SBATCH --output=mega_gpu66-%j.out
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

#python process_ultrafine.py
#cd ../

#data_dir="data/processed_data"
#output_dir="output"
#device=0

#python3 lite.py --data_dir "data/processed_data" \
#                             --output_dir "output" \
#                             --train_batch_size 4 \
#                             --num_train_epochs 1 \
#                             --margin 0.1 \
#                             --save_epochs 1 \
#                             --learning_rate 1e-6 \
#                             --lamb 0.05

data_dir="data/processed_data"
output_dir="output"
device=0

python3 lite.py --data_dir "data/processed_data" \
                             --output_dir "output" \
                             --train_batch_size 32 \
                             --num_train_epochs 42 \
                             --margin 0.1 \
                             --save_epochs 1 \
                             --learning_rate 1e-6 \
                             --lamb 0.05 \
                             --resume_from_checkpoint model_dir/big_model_24_epoch/model \
                             --resume_epoch 24 \
                             --resume_step 0

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

#python3 result.py --dev "./model_dir/Evaluation_dev_processed.json" \
#                   --test "./model_dir/Evaluation_test_processed.json" \
#                   --model_dir "./model_dir/" \
#                   --threshold_step 0.05

#wget http://nlp.cs.washington.edu/entity_type/data/ultrafine_acl18.tar.gz

#tar -xvzf ultrafine_acl18.tar.gz

#python run_code.py 

echo "== End of Job =="