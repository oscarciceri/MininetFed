import paho.mqtt.client as mqtt
import time

BROKER_ADDRESS = "seu_endereco_broker"
BROKER_PORT = "seu_porta_broker"
TOPIC_BYTES_RECEIVED = "$SYS/broker/bytes/received"
TOPIC_BYTES_SENT = "$SYS/broker/bytes/sent"
OUTPUT_FILE = "/flw/broker_info.txt"

def on_message_received(client, userdata, message):
    with open(OUTPUT_FILE, 'a') as f:
        f.write(f"Bytes Recebidos: {str(message.payload.decode('utf-8'))}\n")

def on_message_sent(client, userdata, message):
    with open(OUTPUT_FILE, 'a') as f:
        f.write(f"Bytes Enviados: {str(message.payload.decode('utf-8'))}\n")

client = mqtt.Client("MQTT")
client.connect(BROKER_ADDRESS, BROKER_PORT)

client.message_callback_add(TOPIC_BYTES_RECEIVED, on_message_received)
client.message_callback_add(TOPIC_BYTES_SENT, on_message_sent)

client.subscribe(TOPIC_BYTES_RECEIVED)
client.subscribe(TOPIC_BYTES_SENT)

client.loop_start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Terminando...")
    client.loop_stop()
