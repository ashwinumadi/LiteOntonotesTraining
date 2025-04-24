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
cd /scratch/alpine/asum8093/LiteOntonotesTraining
conda activate py38-pt1131-cuda117

echo "== This is the scripting step! =="

#pip install datasets

wget http://nlp.cs.washington.edu/entity_type/data/ultrafine_acl18.tar.gz

tar -xvzf ultrafine_acl18.tar.gz

#python run_code.py 

echo "== End of Job =="