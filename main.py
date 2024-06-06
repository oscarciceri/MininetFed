from federated import FedNetwork

import sys
import signal

def ctrl_c_handler(signum, frame, fed_network_instance):
    print("\nCtrl+C: Cleaning MininetFed...")
    fed_network_instance.interrupt_execution()
    print("Exiting...")

# Registra o manipulador de sinal para SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, ctrl_c_handler)

def run():
  
  n = len(sys.argv)
  if (n < 2):
      print("correct use: sudo python3 main.py <config.yaml> ...")
      exit()

  for CONFIGYAML in sys.argv[1:]:
    f = FedNetwork(CONFIGYAML)
    signal.signal(signal.SIGINT, lambda sig, fr: ctrl_c_handler(sig, fr, f)) # Ele dá um bad file descriptor no Stop por algum motivo
    f.start()
  


if __name__ == "__main__":
  run()