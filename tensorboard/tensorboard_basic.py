# %%
import numpy as np
import json
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

start = time.time()

# %%
writer = SummaryWriter('logs/run3')

config_param = 5

config_dict = {
  "tfds_training_data": {
      "name": "mnist",
      "split": "train",
      "shuffle_files": "True",
  },

  "keras_optimizer": {
      "name": "Adagrad",
      "learning_rate": "0.001",
      "epsilon": 1e-07,
  },

  "hardware": "Local 1650 Ti",
}

writer.add_text('config', '<pre>'+json.dumps(config_dict, indent=4)+'</pre>', 0)   
log_time_list = []
calc_time_list = []

for i in tqdm(range(10**5)):
    t = time.time()
    a = -np.cos(i)*(i**2)-i**2 
    b = np.random.rand()
    c = np.exp(i**(0.001 )) 
    calc_time_list.append(time.time() - t)
    
    s = time.time()
    writer.add_scalar('scalars/a', a, i)
    writer.add_scalar('scalars/b', b, i)
    writer.add_scalar('scalars/c', c, i)   
    log_time_list.append(time.time()-s)
    # print(i)

writer.close()

print(f'Avg Calc Time = {np.array(calc_time_list).mean()}')
print(f'Avg Log Time = {np.array(log_time_list).mean()}')
print(f'Total Time = {time.time() - start}')

