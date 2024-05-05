import optuna
import subprocess 
import os
import tensorflow as tf
import numpy as np


path_to_file = '/home/rl/OmniIsaacGymEnvs/omniisaacgymenvs/scripts/rlgames_train.py'



command_1 = 'alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh'
python_path = '/home/rl/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh'

n_trials=300

max_iter = 300

r0_scale_min=1
r0_scale_max=15

r_scale_min = 1
r_scale_max=50

gr_scale_min=1
gr_scale_max=100

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



def objective(trial):
    #r0_scale=trial.suggest_int('r0_scale', r0_scale_min,r0_scale_max)
    #r_scale = trial.suggest_int('r_scale', r_scale_min,r_scale_max)
    #gr_scale = trial.suggest_int('gr_scale', gr_scale_min,gr_scale_max)
    
    r0_scale=trial.suggest_int('r0_scale', r0_scale_min,r0_scale_max)
    r1_scale=trial.suggest_int('r1_scale', r0_scale_min,r0_scale_max)
    r2_scale=trial.suggest_int('r2_scale', r0_scale_min,r0_scale_max)

    l_scale=trial.suggest_int('l_scale', r_scale_min,r_scale_max)
    r3_scale=trial.suggest_int('r3_scale', gr_scale_min,gr_scale_max)
    r4_scale=trial.suggest_int('r4_scale', gr_scale_min,gr_scale_max)

    scales = [r0_scale, r1_scale, r2_scale, l_scale, r3_scale, r4_scale, r4_scale]

    experiment_name = 'act_scales_{}_{}_{}_{}_{}_{}_{}'.format(r0_scale, r1_scale, r2_scale, l_scale, r3_scale, r4_scale, r4_scale)
    
    #max_iterations={} task.env.actionScale=[{}, {},{},{},{},{},{}]
    command_2 = '{} {} task=ForwarderPick headless=True experiment={} max_iterations={}  task.env.actionScale=[{},{},{},{},{},{},{}]'.format(
                                                                                                          python_path, 
                                                                                                          path_to_file, 
                                                                                                          #max_iter, 
                                                                                                          experiment_name,
                                                                                                          max_iter,
                                                                                                          r0_scale, r1_scale, r2_scale, l_scale, r3_scale, r4_scale, r4_scale
                                                                                                          )
    #task.env.max_epochs={} experiment={} , experiment_name, 2
    
    command = '{} ; {} >/dev/null 2>&1'.format(command_1, command_2)
    #command = '{} ; {} '.format(command_1, command_2)
    
    #result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    os.system(command=command)
    #print(result)
    
    path_to_log_folder = '/home/rl/OmniIsaacGymEnvs/omniisaacgymenvs/runs/{}/summaries'.format(experiment_name)

    event_acc = EventAccumulator(path_to_log_folder)
    event_acc.Reload()

    # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    reward = []
    for event in event_acc.Scalars('rewards/step'):
        reward.append(event.value)
    reward = np.array(reward)
    
    return np.mean(reward)
    #return np.median(reward)
    #return np.max(reward)

if __name__ == '__main__':
    
    print('CUDA AVAILABLE: ', tf.test.is_gpu_available())
    study = optuna.create_study(direction='maximize', 
                                sampler=optuna.samplers.RandomSampler(),
                                pruner=optuna.pruners.MedianPruner()
                                )
    study.optimize(objective, n_trials=n_trials)
    
    print(study.best_trial)


    #print(result)

