import paho.mqtt.client as mqtt
import sys
import os

os.umask(0o000)

n = len(sys.argv)
if (n != 3):
    print("correct use: python network_monitor.py <broker_address> <output.net>.")


class color:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD_START = '\033[1m'
    BOLD_END = '\033[0m'
    RESET = "\x1B[0m"

  
BROKER_ADDR = sys.argv[1]
FILE_NAME  = sys.argv[2]


# Callback quando o cliente recebe uma resposta CONNACT do servidor.
def on_connect(client, userdata, flags, rc):
    print("Conectado com o código de resultado "+str(rc))
    client.subscribe("$SYS/broker/bytes/#")
    client.subscribe('minifed/stopQueue')

# Callback quando uma mensagem PUBLISH é recebida do servidor.
def on_message(client, userdata, msg):
    tipo = ""
    if 'sent' in msg.topic:
        tipo = "sent:"
    else:
        tipo = "recived:"
    
    with open(FILE_NAME, 'a') as f:
        f.write(f'{tipo} {str(msg.payload.decode("utf-8"))}\n')


def on_message_stop(client, userdata, message):
    print(color.RED + f'received message to stop!')
    exit()


client = mqtt.Client()
client.on_connect = on_connect
client.message_callback_add('minifed/stopQueue', on_message_stop)
client.on_message = on_message

client.connect(BROKER_ADDR, bind_port=1883)

client.loop_forever()  # Bloqueie a chamada de rede
