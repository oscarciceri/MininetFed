from federated import FedNetwork

import sys
import signal
import os


def ctrl_c_handler(signum, frame, fed_network_instance):
    print("\nCtrl+C: Cleaning MininetFed...")
    fed_network_instance.interrupt_execution()
    print("Exiting...")


# Registra o manipulador de sinal para SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, ctrl_c_handler)


def run():
    n = len(sys.argv)
    if n < 2:
        print("correct use: sudo python3 main.py <config.yaml> ...")
        exit()

    files_to_process = []

    for path in sys.argv[1:]:
        if os.path.isdir(path):
            files_to_process.extend([os.path.join(path, f)
                                    for f in os.listdir(path) if f.endswith('.yaml')])
        elif os.path.isfile(path):
            files_to_process.append(path)
        else:
            print(
                f"Warning: {path} is neither a file nor a directory. Ignoring it.")

    if not files_to_process:
        print("No valid .yaml files found.")
        exit()

    for CONFIGYAML in files_to_process:
        f = FedNetwork(CONFIGYAML)
        signal.signal(signal.SIGINT, lambda sig,
                      fr: ctrl_c_handler(sig, fr, f))
        f.start()


if __name__ == "__main__":
    run()
