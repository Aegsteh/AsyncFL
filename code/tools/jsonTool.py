import json
import os,sys
os.chdir(sys.path[0])

# read a json file and convert it to a python dict

def generate_config(json_file):
    json_path = os.path.join('../config',json_file)      # get json file path
    with open(json_path) as f:
        config = json.load(f)
    return config

def print_config(config_dict):
    for key, value in config_dict.items():
        print("- {} : {}".format(key,value))

def get_config_file(mode):
    # get config file about config
    json_file_name = ''
    if mode == 'FedBuff':
        json_file_name = 'FedBuffConfig.json'
    json_path = os.path.join('../config', json_file_name)
    return json_path