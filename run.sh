export MLFLOW_EXPERIMENT_NAME=baseline-1 && python main.py --device cuda --num_workers 5 --batch_size 100 --exp_name 3rd --max_epochs 50 --dropout 0.3 --learning_rate 1e-5

export MLFLOW_EXPERIMENT_NAME=baseline-1 && python main.py --device cuda --num_workers 5 --batch_size 100 --exp_name 4th --max_epochs 50 --dropout 0.3 --learning_rate 1e-6

export MLFLOW_EXPERIMENT_NAME=baseline-1 && python main.py --device cuda --num_workers 10 --batch_size 32 --exp_name 5th --max_epochs 100 --dropout 0.5 --learning_rate 5e-6

export MLFLOW_EXPERIMENT_NAME=baseline-1 && python main.py --device cuda --num_workers 10 --batch_size 32 --exp_name 6th --max_epochs 100 --dropout 0.5 --learning_rate 5e-5 --fc_size 1024