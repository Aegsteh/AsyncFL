import json
import os,sys
os.chdir(sys.path[0])

# read a json file and convert it to a python dict

def generate_config(json_file):
    json_path = os.path.join('../config',json_file)      # get json file path
    with open(json_path) as f:
        config = json.load(f)

    return config