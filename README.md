# 🌀 Avaliação de Nível

> Documentação necessário para a solução proposta da avaliação técnica da Mobit. 

## 📋 Descrição

Este projeto é desenvolvido em **Python 3.12** e utiliza o [`uv`](https://github.com/astral-sh/uv) como gerenciador de pacotes. O `uv` oferece desempenho superior e gerenciamento simplificado de dependências, substituindo ferramentas tradicionais como `pip`, `virtualenv` e `conda`.

## 🏗️ Estrutura de Diretórios

mobit_challenge/
├── data/            # Dados de entrada
├── models/          # Pesos treinados das redes
├── results/         # Gráficos e métricas geradas
├── results/
    ├── parte_1

├── src/             # Código-fonte (funções, classes)
├── utils/           # Funções utilitárias e helpers
├── part_1.py        # Script parte 1
├── part_2.py        # Script parte 2
├── part_3.py        # Script parte 3
├──
├── requirements.txt
└── README.md


## 🧱 Estrutura da Avaliação

O projeto está dividido nas seguintes etapas:

### 🔹 Parte 1 – Processamento Digital de Imagens
Algoritmos de pré-processamento, transformações e análise de imagens.

### 🔹 Parte 2 – Contagem de Pessoas com YOLOv8
Uso do YOLOv8 para contagem de pessoas a partir da deteção.

### 🔹 Parte 3 – Classificador de Tipos de Carros
Aplicação prática de um modelo treinado para identificação e classificação de veículos a partir de imagens.

## 🛠️ Requisitos

- Python >= 3.11
- `uv` instalado globalmente
- CUDA Toolkit instalado (opcional, mas recomendado para para aceleração por GPU)
  - Versão do CUDA = 11.8

### Como instalar o `uv`

```bash
curl -Ls https://astral.sh/uv/install.sh | bash 
```

## ⚙️ Como rodar o projeto

```
uv venv
```

```
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

```
uv pip install -r requirements.txt
```

Somente após instalar as dependências, instale separadamente o torch baseado na versao do seu CUDA
```
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```