import json
import os,sys
os.chdir(sys.path[0])

# read a json file and convert it to a python dict

def generate_config(json_path):
    # json_path = os.path.join('../config',json_file)      # get json file path
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
    elif mode =='period':
        json_file_name = 'PeriodConfig.json'
    elif mode == 'sync':
        json_file_name = 'SyncConfig.json'
    elif mode == 'afo':
        json_file_name = 'AfoConfig.json'
    elif mode == 'crafl':
        json_file_name = 'CRAFLConfig.json'
    json_path = os.path.join('../config', json_file_name)
    return json_path