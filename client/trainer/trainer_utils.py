import os


def read_energy():
    """
    Lê um valor float de um arquivo.

    :param file_path: Caminho para o arquivo.
    :return: Valor float lido ou None se houver erro.
    """

    # Caminho do arquivo de energia
    file_path = "../tmp/consumption"

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"O arquivo {file_path} não foi encontrado.")

        with open(file_path, 'r') as file:
            content = file.read().strip()

        # Tenta converter o valor para float
        value = float(content)
        return value
    except ValueError:
        print(f"Erro: O valor no arquivo {file_path} não é um float válido.")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Erro inesperado: {e}")
    return None
