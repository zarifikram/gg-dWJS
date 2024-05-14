train_steps=2500


python run_hydra.py method=db_egfn n_train_steps=$train_steps replay_sample_size=16 seed=0 &
python run_hydra.py method=db_egfn n_train_steps=$train_steps replay_sample_size=16 feedback=True seed=0 &
python run_hydra.py method=db n_train_steps=$train_steps replay_sample_size=16 seed=0 &
python run_hydra.py method=sac n_train_steps=$train_steps replay_sample_size=16 seed=0 &
python run_hydra.py method=tb augmented=true n_train_steps=$train_steps replay_sample_size=16 seed=0 &
python run_hydra.py method=db_egfn n_train_steps=$train_steps replay_sample_size=16 random_policy=True seed=0 &
python run_hydra.py method=db_egfn n_train_steps=$train_steps replay_sample_size=16 seed=1 &
python run_hydra.py method=db_egfn n_train_steps=$train_steps replay_sample_size=16 feedback=True seed=1 &
python run_hydra.py method=db n_train_steps=$train_steps replay_sample_size=16 seed=1 &
python run_hydra.py method=sac n_train_steps=$train_steps replay_sample_size=16 seed=1 &
python run_hydra.py method=tb augmented=true n_train_steps=$train_steps replay_sample_size=16 seed=1 &
python run_hydra.py method=db_egfn n_train_steps=$train_steps replay_sample_size=16 random_policy=True seed=1 &
python run_hydra.py method=db_egfn n_train_steps=$train_steps replay_sample_size=16 seed=2 &
python run_hydra.py method=db_egfn n_train_steps=$train_steps replay_sample_size=16 feedback=True seed=2 &
python run_hydra.py method=db n_train_steps=$train_steps replay_sample_size=16 seed=2 &
python run_hydra.py method=sac n_train_steps=$train_steps replay_sample_size=16 seed=2 &
python run_hydra.py method=tb augmented=true n_train_steps=$train_steps replay_sample_size=16 seed=2 &
python run_hydra.py method=db_egfn n_train_steps=$train_steps replay_sample_size=16 random_policy=True seed=2 &
wait


