
#%%
import json
from typing import Dict


def save_to_json(path: str, data: Dict) -> None:
    """"""

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



import yaml

with open('fut_conf.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
# %%
