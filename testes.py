import json
import numpy as np

# SERVER novos weights
agg_weights = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])




response = json.dumps({'weights': [w.tolist() for w in agg_weights]})
json_enviado = response


# CLIENT recebendo atualizado
msg = json.loads(json_enviado)
agg_weights = [np.asarray(w, dtype=np.float32) for w in msg["weights"]]


# CLIENT selecionado e enviando de volta

response = json.dumps({'id': 1, 'weights': [w.tolist(
) for w in agg_weights], 'num_samples': 1})
json_enviado = response



# Server recebendo weights treinados do cliente
m = json.loads(json_enviado)
weights = [np.asarray(w, dtype=np.float32) for w in m['weights']]

print(weights[1])