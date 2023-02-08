

import numpy as np
import time


sample_list1 = list(range(1000000))
sample_list2 = list(range(1000000))


start = time.time()
combined_sum = [num1 + num2 for num1, num2 in zip(sample_list1, sample_list2)]
end = time.time()
print(end - start)


sample_array1 = np.arange(1000000)
sample_array2 = np.arange(1000000)

start = time.time()
combined_sum = sample_array1 + sample_array2
end = time.time()
print(end - start)