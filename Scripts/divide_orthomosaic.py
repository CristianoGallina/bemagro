import os
import argparse
import rasterio
from rasterio.windows import Window
from rasterio.plot import reshape_as_image
import numpy as np
from PIL import Image

def split_image(input_path, output_dir, block_size=256):
    # Criar o diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Abrir a imagem com rasterio
    with rasterio.open(input_path) as dataset:
        width = dataset.width
        height = dataset.height
        channels = dataset.count  # Número de canais na imagem

        print(f"Largura: {width}, Altura: {height}, Canais: {channels}")

        # Iterar sobre blocos
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                # Definir a janela de leitura
                window = Window(j, i, block_size, block_size)
                transform = dataset.window_transform(window)

                # Ler o bloco da imagem
                block = dataset.read(window=window)

                # Reshape para H x W x C (imagem multicanal)
                block_image = reshape_as_image(block)

                # Salvar como PNG
                block_path = os.path.join(output_dir, f"block_{i}_{j}.png")
                Image.fromarray(block_image).save(block_path)

    print(f"Imagem dividida e salva em: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dividir ortomosaico em blocos menores.")
    parser.add_argument("--input", required=True, help="Caminho do arquivo de ortomosaico (TIFF ou outro formato).")
    parser.add_argument("--output", required=True, help="Caminho do diretório onde as imagens divididas serão salvas.")
    parser.add_argument("--block_size", type=int, default=256, help="Tamanho do bloco (padrão: 256).")

    args = parser.parse_args()
    split_image(args.input, args.output, args.block_size)