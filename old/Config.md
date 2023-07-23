Config.json 

[ -- VARIOS ( TESTE)

    “RD”: 200, 

    “QTD”: 10, 

    “SDEV”: [ 

        { 

        mem: 20,
        cpu:  

        }  

    ],  

] 

 

    Root (Objeto raiz) – Tipo Array ou Objeto 

        Array para várias topologias e objeto quando única topologia 

    RD  “rounds” - inteiro 

        Quantidade de rodadas 

    QTD “Quantidade de dispositivos” -  Quantidade de dispositivos 

    DPID Id de partição de dados - String 
        - USUARIOS (8 dados unicos) - 8 device.
        - Esporte 5 , Usuario (Pre processamento)
            - Futebol 1M , Volei 1C, (balanceado) 1000000100 (20) 
            - Id pro index ( 1 - 1000, 1001-2000)
   
        Indica qual coluna ID de partição dos dados, coluna para separação dos dados de cada dispositivo, caso preenchida ignora o campo quantidade de dispositivos, a quantidade passa a ser quantidade de dados únicos na coluna ID.  

    SDEV – “Estrutura de hardware dos dispositos” array 
        MEM -> Quantidade em Mb de memória. 
        CPU -> Nucleo
        CPU -> Fc

 