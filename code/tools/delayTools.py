'''
simulate delay
'''


def generate_delays(global_config):
    # delay config, including mode, params
    delay_config = global_config["delay"]
    n_clients = global_config["n_clients"]          # the number of clients
    mode = delay_config["mode"]
    param = delay_config["param"]
    delays = []
    if mode == "base":
        for i in range(n_clients):
            # delays.append(param * 5 * i + 1)
            delays.append(2.4 * param + i)
    return delays


def generate_bandwiths(global_config):
    # delay config, including mode, params
    bandwith_config = global_config["bandwith"]
    n_clients = global_config["n_clients"]          # the number of clients
    mode = bandwith_config["mode"]
    bandwith0 = bandwith_config["param"]
    bandwiths = []
    if mode == "base":
        for i in range(n_clients):
            bandwiths.append(bandwith0 / (i + 1))
    return bandwiths
