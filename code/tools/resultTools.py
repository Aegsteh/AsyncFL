import os,sys
import json

os.chdir(sys.path[0])

def save_to_file(path,result):
    with open(path,'w') as f:
        for i in result:
            f.write(i+'\n')

def save_results(root=None,dir_name=None,config=None,global_loss=None,global_acc=None,staleness=None):
    dir_root = os.path.join('../results', root)          # experiment path
    dir_path = os.path.join(dir_root, dir_name) 

    if not config is None:      # write config to json
        config_file_name = 'config.json'
        config_file_path = os.path.join(dir_path,config_file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(config_file_path, 'w') as f:
            json.dump(config, f)
    
    if not global_loss is None:
        global_loss_name = "global_loss.txt"
        global_loss_path = os.path.join(dir_path,global_loss_name)
        save_to_file(global_loss_path,global_loss)
    
    if not global_acc is None:
        global_acc_name = "global_loss.txt"
        global_acc_path = os.path.join(dir_path,global_acc_name)
        save_to_file(global_acc_path,global_acc)
    
    if not staleness is None:
        staleness_name = "staleness.txt"
        staleness_path = os.path.join(dir_path,staleness_name)
        save_to_file(staleness_path,staleness)