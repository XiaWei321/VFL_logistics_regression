from ClientA import ClientA
from ClientB import ClientB
from ClientC import ClientC

def process_A(name, data, config):
    client_A = ClientA(name, data, config)
    client_A.set_up()
    client_A.connect([config['ADDR_B'], config['ADDR_C']])