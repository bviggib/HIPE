import os
import torch
import pandas as pd
from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.core.data import Data
from ax.modelbridge.registry import Models

from yahpo_gym import BenchmarkSet
from yahpo_gym.local_config import local_config

from experiments.lcbench import get_lcbench_experiment, get_lcbench_surrogate

root = '/Users/viggy/Documents/GitHub/HIPE'
yahpo_root = os.path.join(root, 'data/yahpo/yahpo_data')
out_dir = os.path.join(root, 'data/lcbench')
os.makedirs(out_dir, exist_ok=True)

if not local_config.settings_path.exists():
  local_config.init_config(data_path=yahpo_root)
local_config.set_data_path(yahpo_root)

lb = torch.tensor([16.0, 0.0, 64.0, 1.0, 1e-4, 0.1, 1e-5], dtype=torch.double)
ub = torch.tensor([512.0, 1.0, 1024.0, 4.0, 1e-1, 0.99, 1e-1], dtype=torch.double)

def sample_points(n: int, seed: int) -> torch.Tensor:
  torch.manual_seed(seed)
  X = lb + (ub - lb) * torch.rand(n, 7, dtype=torch.double)
  X[:, 0] = X[:, 0].round().clamp(16, 512)
  X[:, 2] = X[:, 2].round().clamp(64, 1024)
  X[:, 3] = X[:, 3].round().clamp(1, 4)
  return X

def build_one(display_name: str, openml_task_id: str, n: int = 192, seed: int = 0):
  bench = BenchmarkSet('lcbench', instance=openml_task_id)
  X = sample_points(n=n, seed=seed)
  vals = []
  for i in range(n):
    cfg = {
      'OpenML_task_id': openml_task_id,
      'epoch': 52,
      'batch_size': int(X[i,0].item()),
      'max_dropout': float(X[i,1].item()),
      'max_units': int(X[i,2].item()),
      'num_layers': int(X[i,3].item()),
      'learning_rate': float(X[i,4].item()),
      'momentum': float(X[i,5].item()),
      'weight_decay': float(X[i,6].item()),
    }
    vals.append(float(bench.objective_function(cfg)[0]['val_accuracy']))

  exp = get_lcbench_experiment(display_name)
  arms = []
  rows = []
  for i in range(n):
    params = {
      'batch_size': int(X[i,0].item()),
      'max_dropout': float(X[i,1].item()),
      'max_units': int(X[i,2].item()),
      'num_layers': int(X[i,3].item()),
      'learning_rate': float(X[i,4].item()),
      'momentum': float(X[i,5].item()),
      'weight_decay': float(X[i,6].item()),
    }
    arm = Arm(parameters=params, name=f'a{i}')
    arms.append(arm)

  trial = exp.new_batch_trial(generator_run=GeneratorRun(arms=arms))
  trial.mark_running(no_runner_required=True)
  trial.mark_completed()

  for i, arm in enumerate(arms):
    rows.append({
      'arm_name': arm.name,
      'metric_name': 'Train/val_accuracy',
      'mean': vals[i],
      'sem': 0.0,
      'trial_index': trial.index,
    })

  data = Data(df=pd.DataFrame(rows))
  exp.attach_data(data)
  mb = Models.BOTORCH_MODULAR(
    surrogate=get_lcbench_surrogate(),
    experiment=exp,
    search_space=exp.search_space,
    data=data,
  )
  obj = {
    'experiment': exp,
    'data': data,
    'state_dict': mb.model.surrogate.model.state_dict(),
  }
  out = os.path.join(out_dir, f'lcbench_{display_name}.pt')
  torch.save(obj, out)
  print('saved', out, 'n=', n)

  # Validation: ensure current repo loader can restore and predict.
  from experiments.lcbench import get_surrogate_and_datasets
  mb2, ds = get_surrogate_and_datasets(out)
  test_pred = mb2.model.predict(torch.rand(3,7,dtype=torch.double))[0]
  print('validated', display_name, 'datasets=', len(ds), 'pred_shape=', tuple(test_pred.shape))

build_one('Australian', '167104', n=192, seed=0)
build_one('car', '189905', n=192, seed=1)
print('done')