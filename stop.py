import paho.mqtt.client as mqtt
import sys
import time

n = len(sys.argv)
if (n != 2):
    print("correct use: python stop.py <broker_address>.")


class color:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD_START = '\033[1m'
    BOLD_END = '\033[0m'
    RESET = "\x1B[0m"


BROKER_ADDR = sys.argv[1]


# Callback quando o cliente recebe uma resposta CONNECT do servidor.
def on_connect(client, userdata, flags, rc):
    client.subscribe('minifed/stopQueue')
    client.subscribe('minifed/autoWaitContinue')


def on_message_stop(client, userdata, message):
    print(color.RED + f'received message to stop!')
    print(color.RESET)
    time.sleep(5)
    client.disconnect()
    exit()


def on_message_continue(client, userdata, message):
    print(color.GREEN + f'received message to continue!')
    print(color.RESET)
    client.disconnect()
    # time.sleep(5)
    exit()


client = mqtt.Client()
client.on_connect = on_connect
client.message_callback_add('minifed/stopQueue', on_message_stop)
client.message_callback_add('minifed/autoWaitContinue', on_message_continue)
try:
    client.connect(BROKER_ADDR, bind_port=1883, keepalive=0)
except:
    pass

print(color.YELLOW + "Waiting...")
client.loop_forever()  # Bloqueie a chamada de rede
