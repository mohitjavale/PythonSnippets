import wandb
import numpy as np
import time
from tqdm import tqdm

s1 = time.time()

config_dict={
    "config_param0": 0.01,
    "config_param1": "test1",
    "config_param2": "test",
    "config_param3": 10,
    }

wandb.init(
    project="test",
    name='run2', 
    config=config_dict
)

# wandb.config = config_dict

time_list = []
for i in tqdm(range(10**5)):
    s2 = time.time()
    a = -np.cos(i)*(i**2)-i**2
    b = np.random.rand()
    c = np.exp(i**(0.001)) 
    log_dict = {"a": b, "b": b, "c":b}
    wandb.log(log_dict)
    # time.sleep(0.002)
    time_list.append(time.time() - s2)


print(f'Avg Loop Time = {np.array(time_list).mean()}')
print(f'Total Time = {time.time() - s1}')

s3 = time.time()
wandb.finish()

print(f'Time for wandb.finish = {time.time() - s3}')
