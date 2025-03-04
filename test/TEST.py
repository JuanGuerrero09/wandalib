from dotenv import load_dotenv
import os
import pywanda
from wandalib import get_transient_results
import matplotlib.pyplot as plt

load_dotenv()
cwd =  os.getenv('WANTA_TEST_DIR')
wanda_bin = os.getenv('WANDA_BIN') 

wanda_name = r'Testmodel.wdi'
wanda_file = os.path.join(cwd, wanda_name)
wanda_model = pywanda.WandaModel(wanda_file, wanda_bin)
wanda_model.reload_output()
print(get_transient_results(wanda_model, ['PIPE LINE MOMRAH 1']))
# print(get_transient_results(wanda_model, ['PIPE LINE START 1', 'PIPE LINE START 2', 'PIPE LINE START 3', 'PIPE LINE START 4']))
plt.show()
# print(dir(wandalib))