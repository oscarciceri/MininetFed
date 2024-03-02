from federated import FedNetwork

import sys

def run():
  
  n = len(sys.argv)
  if (n < 2):
      print("correct use: sudo python3 run.py <config.yaml> ...")
      exit()

  for CONFIGYAML in sys.argv[1:]:
    f = FedNetwork(CONFIGYAML)
    f.start()
  


if __name__ == "__main__":
  run()