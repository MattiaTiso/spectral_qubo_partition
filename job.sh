#!/bin/bash

#SBATCH --job-name=quboablocchi
#SBATCH --partition=a-gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --exclusive

#SBATCH --output=logs/quboablocchi_%j.out
#SBATCH --error=logs/quboablocchi_%j.err

#SBATCH --nodelist=acnode03
#SBATCH --qos=only-gpu
#SBATCH --gres=gpu:1

#SBATCH --account=eupuser

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


module purge
module load nvidia/cuda-12.5
echo "Start: $(date)"

# Monitor GPU ogni secondo
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
           --format=csv -l 1 > gpu_usage.log &
GPU_MON_PID=$!

source /home/mattia.tiso/qubovenv/bin/activate

cd /home/mattia.tiso/quboproj || exit 1

python3 demo_quboablocchi.py

# Ferma il monitor
kill $GPU_MON_PID


echo "End: $(date)"