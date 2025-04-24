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

#python process_ultrafine.py
#cd ../
data_dir="data/small_processed_data"
output_dir="small_output"
device=0

python3 lite.py --data_dir "data/small_processed_data" \
                             --output_dir "small_output" \
                             --train_batch_size 4 \
                             --num_train_epochs 100 \
                             --margin 0.1 \
                             --save_epochs 10 \
                             --learning_rate 1e-6 \
                             --lamb 0.05

#wget http://nlp.cs.washington.edu/entity_type/data/ultrafine_acl18.tar.gz

#tar -xvzf ultrafine_acl18.tar.gz

#python run_code.py 

echo "== End of Job =="