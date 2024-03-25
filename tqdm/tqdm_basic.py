from tqdm import tqdm
import time

t = tqdm(range(1000))
for i in t:
    j = i**2
    t.set_description(f'{j=}')
    t.set_postfix(j=j, key=j**0.25)
    time.sleep(0.1)