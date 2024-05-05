import optuna
import subprocess 
import os
import tensorflow as tf
import numpy as np
from decimal import Decimal
import pandas as pd
path_to_file = '/home/rl/OmniIsaacGymEnvs/omniisaacgymenvs/scripts/rlgames_train.py'



command_1 = 'alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh'
python_path = '/home/rl/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh'

n_trials=300

max_iter = 1000

stiffness_min = 1e3
stiffness_max = 1e15

force_min = 1e3
force_max = 1e20

n_joints = 7

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


csv_path = 'gains.csv'
init_gains_dict = {}
init_gains_dict['trial_no'] = []
for j in range(n_joints):
    init_gains_dict['s{}'.format(j)] = []
    init_gains_dict['f{}'.format(j)] = []
init_gains_dict['result'] = []

if(not os.path.exists(csv_path)):
    df = pd.DataFrame(init_gains_dict)
    df.to_csv(csv_path)

def objective(trial):

    '''
    s0 = trial.suggest_int('s0_scale', stiffness_min, stiffness_max, step=5e3)
    s1 = trial.suggest_int('s1_scale', stiffness_min, stiffness_max, step=5e3)
    s2 = trial.suggest_int('s2_scale', stiffness_min, stiffness_max, step=5e3)
    s3 = trial.suggest_int('s3_scale', stiffness_min, stiffness_max, step=5e3)
    s4 = trial.suggest_int('s4_scale', stiffness_min, stiffness_max, step=5e3)
    s5 = trial.suggest_int('s5_scale', stiffness_min, stiffness_max, step=5e3)
    s6 = trial.suggest_int('s6_scale', stiffness_min, stiffness_max, step=5e3)
    '''

    kps = []
    force = []

    gains_dict = init_gains_dict.copy()
    gains_dict['trial_no'] = trial.number
    for j in range(n_joints):
        s = trial.suggest_int('s{}_scale'.format(j), stiffness_min, stiffness_max, step=1e3)
        f = trial.suggest_int('f{}_scale'.format(j), force_min, force_max, step=1e3)
        kps.append(s)
        force.append(f)

        init_gains_dict['s{}'.format(j)] = [s]
        init_gains_dict['f{}'.format(j)] = [f]
        
    '''
    f0 = trial.suggest_int('f0_scale', force_min, force_max, step=5e3)
    f1 = trial.suggest_int('f1_scale', force_min, force_max, step=5e3)
    f2 = trial.suggest_int('f2_scale', force_min, force_max, step=5e3)
    f3 = trial.suggest_int('f3_scale', force_min, force_max, step=5e3)
    f4 = trial.suggest_int('f4_scale', force_min, force_max, step=5e3)
    f5 = trial.suggest_int('f5_scale', force_min, force_max, step=5e3)
    f6 = trial.suggest_int('f6_scale', force_min, force_max, step=5e3)
    '''

    #experiment_name = 's[{},{},{},{},{},{},{}]_f[{},{},{},{},{},{},{}]'.format(s0, s1, s2, s3, s4, s5, s6,
    #                                                                            f0, f1, f2, f3, f4, f5, f6)
    experiment_name = 'exp_{}'.format(trial.number)
    #max_iterations={} task.env.actionScale=[{}, {},{},{},{},{},{}]
    command_2 = "{} {} task=ForwarderPick headless=True experiment={} max_iterations={} \'task.env.kps={}\' \'task.env.force={}\'".format(
                                                                                                          python_path, 
                                                                                                          path_to_file, 
                                                                                                          #max_iter, 
                                                                                                          experiment_name,
                                                                                                          max_iter,
                                                                                                          kps,
                                                                                                          force
                                                                                                          )

    #task.env.max_epochs={} experiment={} , experiment_name, 2
    command = '{} ; {}'.format(command_1, command_2)
    
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result)
    
    path_to_log_folder = '/home/rl/OmniIsaacGymEnvs/omniisaacgymenvs/runs/{}/summaries'.format(experiment_name)

    event_acc = EventAccumulator(path_to_log_folder)
    event_acc.Reload()

    # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    reward = []
    for event in event_acc.Scalars('rewards/step'):
        reward.append(event.value)
    reward = np.array(reward)
    
    mean_reward = np.mean(reward)
    
    gains_dict['result'] = mean_reward

    current_run_df = pd.DataFrame(gains_dict)

    current_run_df.to_csv(csv_path, mode='a', index=False, header=True)

    
    return mean_reward
    #return np.median(reward)
    #return np.max(reward)

if __name__ == '__main__':
    
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(study.best_trial)


    #print(result)

