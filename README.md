# Classificacao-Bert-em-Reviews-de-E-commerce
projeto implementa um pipeline completo de Processamento de Linguagem Natural (NLP) focado em reviews de produtos no mercado brasileiro. O sistema utiliza **Fine-Tuning** de um modelo BERT pré-treinado para classificação de sentimentos e um pipeline secundário para Extração de Entidades Nomeadas (NER).

📋 Visão Geral

O projeto está dividido em dois módulos principais:

1. **Classificador de Sentimentos**: Treinamento de um modelo para categorizar comentários em Negativo, Neutro ou Positivo, utilizando o dataset B2W.

2. **Reconhecimento de Entidades (NER)**: Extração de nomes de pessoas, organizações e locais a partir de textos não estruturados.

🛠️ **Tecnologias e Dependências**

O código foi desenvolvido em Python e utiliza o ecossistema Hugging Face.

**Principais Bibliotecas:**

``transformers``: Modelos e pipelines de NLP.

``evaluate & sklearn``: Cálculo de métricas e matriz de confusão.

``datasets``: Manipulação eficiente de dados para o Hugging Face.

``pandas & numpy``: Manipulação de dados tabulares.

``seaborn & matplotlib``: Visualização de dados.

Instalação:
``
!pip install evaluate
!pip install --upgrade transformers
!pip install datasets scikit-learn pandas seaborn
``

**📊 Dataset e Pré-processamento**

Fonte dos Dados: ``B2W-Reviews01.csv`` (Dataset público de reviews do grupo B2W).

1. **Limpeza e Transformação**

* **Tratamento de Separadores**: O arquivo é lido ajustando separadores para evitar erros de bad lines.

* **Remoção de Nulos**: Linhas sem texto de review são descartadas.

* **Conversão de Labels**: As notas (estrelas de 1 a 5) são convertidas para sentimentos:

* ⭐ 1 e 2 → 0 (Negativo)

* ⭐ 3 → 1 (Neutro)

* ⭐ 4 e 5 → 2 (Positivo)

2. **Balanceamento de Classes**

Para evitar viés (já que reviews positivos são predominantes), aplicou-se uma sub-amostragem (undersampling):

Selecionados aleatoriamente 1.500 exemplos de cada classe (Negativo, Neutro, Positivo).

Total do dataset de treino/teste: 4.500 exemplos.

Divisão: 80% Treino / 20% Teste.

**🧠 Modelo de Sentimento (Fine-Tuning)**

Utilizou-se o modelo **BERTimbau** ``(neuralmind/bert-base-portuguese-cased)``, o estado da arte para língua portuguesa.

**Configurações de Treinamento**

* **Épocas**: 3

* **Learning Rate**: 2e-5

* **Batch Size**: 16

* **Max Length**: 128 tokens (truncamento para otimização).

* **Otimizador**: AdamW com Weight Decay.

**Resultados Obtidos**

Após o treinamento, o modelo foi avaliado no conjunto de teste (900 amostras):

| Métrica | Valor |
| :--- | :--- |
| **Acurácia Global** | **76.89%** |

**Performance por Classe**:

Negativo: Alta precisão (87%) e Recall (83%). O modelo distingue bem reclamações.

Positivo: Bom equilíbrio (81% Precision / 81% Recall).

Neutro: É a classe mais difícil (63% Precision), frequentemente confundida com positivos ou negativos leves.

O modelo final foi salvo no diretório ``./modelo_b2w_sentimento.``

**🔍 Módulo de NER (Named Entity Recognition)**

Além da classificação, implementou-se um pipeline para extração de entidades usando o modelo ``Davlan/bert-base-multilingual-cased-ner-hrl.``

**Objetivo**: Identificar:

* ``PER``: Pessoas (ex: nomes de atendentes).

* ``ORG``: Organizações (ex: Lojas, Marcas).

* ``LOC``: Locais (ex: Cidades, Estados).

**Exemplo de Saída**:

Texto: "Falei com o atendente João."

* Entidade: João (PER) | Confiança: 99.91%

Texto: "Comprei um iPhone 12 na loja da Vivo em São Paulo."

* Entidade: Vivo (ORG)

* Entidade: São Paulo (LOC)

🚀 **Como Executar**

1. Certifique-se de ter o arquivo B2W-Reviews01.csv no diretório raiz.

2. Execute o script de Carga e Treinamento para gerar o modelo de sentimento.

3. O script gerará automaticamente:

* Métricas no console.

* Uma Matriz de Confusão visual (plotada com Seaborn).

4. Execute o bloco de Inferência NER para testar a extração de entidades em frases arbitrárias.

⚠️ **Notas Importantes**

* **Autenticação HF**: O script remove explicitamente o token do Hugging Face (os.environ.pop('HF_TOKEN')) para evitar conflitos em ambientes públicos/compartilhados.

* **Tratamento de Erros**: Há um bloco try/except na leitura do CSV que cria um DataFrame dummy caso o arquivo não seja encontrado, permitindo testar o código sem o dataset completo.

**Créditos dos Modelos**:

* NeuralMind (BERTimbau)

* Davlan (Multilingual NER)
