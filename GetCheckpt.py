import os
import numpy as np
import re
output_directory = "exported/"

# goes through the model is the training/ dir and gets the last one.
# you could choose a specfic one instead of the last
lst = os.listdir("models/my_mobilenet/")
#print(lst)
lst = [l for l in lst if 'ckpt-' in l and '.index' not in l]
steps=np.array([int(re.findall('\d+', l)[0]) for l in lst])
last_model = lst[steps.argmax()]
last_model_path = os.path.join('models/my_mobilenet', last_model)
print(last_model_path)