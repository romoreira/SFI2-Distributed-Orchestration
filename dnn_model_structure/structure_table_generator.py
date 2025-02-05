import pandas as pd
import torchinfo
from tsai.all import *

# dataset id
dsid = 'NATOPS' 
X, y, splits = get_UCR_data(dsid, return_split=False)
X.shape, y.shape, splits

X_train, y_train, X_test, y_test  = get_UCR_data(dsid, return_split=True)
X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])

tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=0)

# Criar um dicionário para armazenar os detalhes dos modelos
model_details = []

# Lista de modelos que vamos avaliar
models = {
    "FCN": FCN,
    "FCNPlus": FCNPlus,
    "ResNet": ResNet,
    "ResNetPlus": ResNetPlus,
    "ResCNN": ResCNN,
    "TCN": TCN,
    "InceptionTime": InceptionTime,
    "InceptionTimePlus": InceptionTimePlus,
    "XceptionTime": XceptionTime,
    "XceptionTimePlus": XceptionTimePlus,
    "OmniScaleCNN": lambda vars, c: OmniScaleCNN(vars, c, seq_len=dls.len),  # Adicionando seq_len
}

# Iterar sobre os modelos
for name, ModelClass in models.items():
    try:
        # Criar o modelo (verifica se é função lambda para OmniScaleCNN)
        model = ModelClass(dls.vars, dls.c) if isinstance(ModelClass, type) else ModelClass(dls.vars, dls.c)

        # Obter informações detalhadas do modelo
        summary = torchinfo.summary(model, input_size=(1, dls.vars, dls.len), verbose=0)

        # Verificar valores para evitar NoneType errors
        num_params = summary.total_params if summary.total_params is not None else 0
        num_trainable_params = summary.trainable_params if summary.trainable_params is not None else 0
        num_non_trainable_params = num_params - num_trainable_params
        num_layers = len(summary.summary_list) if summary.summary_list else 0

        # Se `total_mult_adds` for None, definir como 0.0
        mult_adds = summary.total_mult_adds
        if mult_adds is None:
            print(f"Aviso: `total_mult_adds` retornou None para {name}. Definindo como 0.")
            mult_adds = 0.0
        else:
            mult_adds /= 1e6  # Converter para milhões (M)

        # Estimativa de tamanho do modelo na memória (MB)
        param_size = num_params * 4 / (1024 ** 2)  # Cada parâmetro em float32 (4 bytes)
        buffer_size = num_trainable_params * 4 / (1024 ** 2)  # Tamanho dos buffers (aproximado)
        total_mem = param_size + buffer_size

        kernel_sizes = [layer.kernel_size for layer in model.modules() if isinstance(layer, torch.nn.Conv1d)]
        pooling_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.AdaptiveAvgPool1d)]
        
        # Obter nomes das camadas
        layer_names = [layer.__class__.__name__ for layer in model.modules()]

        # Salvar no dicionário
        model_details.append({
            "Modelo": name,
            "Parâmetros": num_params,
            "Treináveis": num_trainable_params,
            "Não Treináveis": num_non_trainable_params,
            "Camadas Totais": num_layers,
            "Tamanhos de Kernel": kernel_sizes if kernel_sizes else "N/A",
            "Pooling": "Sim" if pooling_layers else "Não",
            "Mult-Adds (M)": round(mult_adds, 2),
            "Tamanho Estimado (MB)": round(total_mem, 2),
            "Camadas": ", ".join(layer_names)  # Adicionar nomes das camadas
        })

    except Exception as e:
        print(f"Erro ao processar {name}: {e}")

# Criar um DataFrame e exibir
df = pd.DataFrame(model_details)
print(df)

# Salvar a tabela em CSV
df.to_csv("detalhes_modelos.csv", index=False)
