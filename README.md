# 🌀 Avaliação de Nível

> Documentação necessário para a solução proposta da avaliação técnica da Mobit. 

## 📋 Descrição

Este projeto é desenvolvido em **Python 3.12** e utiliza o [`uv`](https://github.com/astral-sh/uv) como gerenciador de pacotes. O `uv` oferece desempenho superior e gerenciamento simplificado de dependências, substituindo ferramentas tradicionais como `pip`, `virtualenv` e `conda`.

## 🧱 Estrutura da Avaliação

O projeto está dividido nas seguintes etapas:

### 🔹 Parte 1 – Processamento Digital de Imagens
Algoritmos de pré-processamento, transformações e análise de imagens.

### 🔹 Parte 2 – Deep Learning
Treinamento e avaliação de modelos de redes neurais profundas com TensorFlow ou PyTorch.

### 🔹 Parte 3 – Classificador de Tipos de Carros
Aplicação prática de um modelo treinado para identificação e classificação de veículos a partir de imagens.

## 🛠️ Requisitos

- Python >= 3.11
- `uv` instalado globalmente
- CUDA Toolkit instalado (opcional, mas recomendado para para aceleração por GPU)

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