'''
simulate delay
'''

def generate_delays(global_config):
    delay_config = global_config["delay"]           # delay config, including mode, params
    n_clients = global_config["n_clients"]          # the number of clients
    mode = delay_config["mode"]
    param = delay_config["param"]
    delays = []
    if mode == "base":
        for i in range(n_clients):
            delays.append(param * (i + 1) / 2)
    return delays