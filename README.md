# Bem Agro
 Desenvolvimento do desafio de IA da Bem Agro como pré-requisito a candidatura de Desenvolvedor De Visão Coputacional.

## Projeto: Vetorização de Máscaras com PyTorch

### Descrição

Este projeto utiliza o PyTorch para realizar a vetorização de máscaras de imagens. O objetivo é processar imagens, aplicar segmentação semântica e gerar representações vetoriais para uso em diferentes aplicações, como análise de dados espaciais, detecção de objetos ou processamento de imagens geoespaciais.

Tecnologias Utilizadas

Python: Linguagem principal do projeto.

PyTorch: Framework de aprendizado de máquina para treinamento e inferência de redes neurais.

Torchvision: Biblioteca complementar para PyTorch, com modelos pré-treinados e ferramentas de visão computacional.

Pillow: Manipulação de imagens.

OpenCV (opcional): Para pré-processamento avançado de imagens.

osgeo/GDAL (opcional): Para manipulação de dados geoespaciais, se necessário.

## Requisitos

Certifique-se de ter as dependências instaladas:

Python 3.8 ou superior

PyTorch

torchvision

Pillow

OpenCV (opcional)

osgeo (opcional)

Instale os pacotes necessários com:

pip install torch torchvision pillow opencv-python gdal

## Estrutura do Projeto

.
01origin/            # Diretório para armazenar imagens de origem 
02orthomozaic/       # Diretório para armazenar imagens geradas pelo script
03binarize/          # Diretório para armazenar imagens binarizadas
04model/             # Diretório para armazenar o modelo treinado
05segmented/         # Diretório para armazenar a máscara segmentada
06polygons/          # Diretório para armazenar polígonos
Scripts/             # Diretório para armazenar os scripts de cada etapa do desafio
README.md            # Documentação do projeto

## Passos para Execução

Prepare os Dados: Coloque as imagens que deseja processar no diretório 01origin/.

1. Execute a quebra da imagem em blocos:
Use o script divide_orthomosaic.py para realizar a quebra da imagem em blocos.

Exemplo:
python scripts/divide_orthomosaic.py --input </01origin/orto.tif> --output </02orthomozaic>

2. Gere o Dataset
Use o script binarize_images.py para binarizar as imagens do orthomozaico.

Exemplo:
python scripts/binarize_images.py --input </02orthomozaic> --output </03binarize>

3. Implemente e treine a rede neural
Use o script train_model.py para treinar a rede neural e salvar o modelo treinado.

Exemplo:
python scripts/train_model.py --rgb </02orthomozaic> --groundtruth </03binarize/> --modelpath </04model/model.h5>

4. Execute a Inferência
Use o script vectorize_mask.py para rodar o modelo de segmentação e gerar máscaras vetorizadas.

Exemplo:
python scripts/vectorize_mask.py --input 01origin/input_image.png --output 05segmented/output_vector.png

Resultados: Os resultados serão salvos no diretório 04segmented/ com as máscaras processadas.

## Personalização

Você pode substituir o modelo padrão por outro modelo treinado em PyTorch. 

A rede neural utilizada é a U-Net. É uma rede neural profunda projetada para segmentação de imagens pixel a pixel. Composta por um encoder (extrai características) e um decoder (restaura resolução), usa conexões para alta precisão. Amplamente usada em geoprocessamento e visão computacional.

## Contato

Dúvidas ou sugestões, entre em contato:

Autor: Cristiano Basso Gallina

LinkedIn: https://www.linkedin.com/in/cristianogallina/
