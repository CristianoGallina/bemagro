import os
import argparse
import numpy as np
from PIL import Image

def calculate_exg(image_array):
    """
    Calcula o índice Excess Green (ExG) para a imagem RGB.

    Args:
        image_array (numpy.ndarray): Array numpy da imagem RGB.

    Returns:
        numpy.ndarray: Array numpy com os valores de ExG.
    """
    R = image_array[:, :, 0].astype(float)  # Canal vermelho
    G = image_array[:, :, 1].astype(float)  # Canal verde
    B = image_array[:, :, 2].astype(float)  # Canal azul

    # Calcular o índice ExG
    exg = 2 * G - R - B
    return exg

def binarize_image_exg(image_path, output_path, threshold=0):
    """
    Binariza uma imagem usando o índice ExG com base no limiar fornecido.

    Args:
        image_path (str): Caminho para a imagem RGB de entrada.
        output_path (str): Caminho para salvar a imagem segmentada.
        threshold (float): Valor de limiar para identificar vegetação (default: 0).
    """
    # Abrir a imagem
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    # Calcular o índice ExG
    exg = calculate_exg(image_array)

    # Criar máscara binarizada com base no limiar
    binary_image = np.zeros_like(exg, dtype=np.uint8)
    binary_image[exg > threshold] = 1

    # Salvar imagem binarizada em escala de cinza
    output_image = Image.fromarray(binary_image * 255)  # Escala 0-255 para salvar como imagem
    output_image.save(output_path)

def process_directory(input_dir, output_dir, threshold=0):
    """
    Processa todas as imagens RGB em um diretório, binarizando-as usando o índice ExG.

    Args:
        input_dir (str): Caminho para o diretório com imagens RGB.
        output_dir (str): Caminho para o diretório de saída das imagens segmentadas.
        threshold (float): Valor de limiar para identificar vegetação (default: 0).
    """
    # Criar diretório de saída, se necessário
    os.makedirs(output_dir, exist_ok=True)

    # Processar cada imagem no diretório
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"binarized_{filename}")
            binarize_image_exg(input_path, output_path, threshold)
            print(f"Imagem processada: {input_path} -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentar imagens RGB para escala de cinza binarizada usando ExG.")
    parser.add_argument("--input", required=True, help="Caminho do diretório que contém as imagens RGB em blocos.")
    parser.add_argument("--output", required=True, help="Caminho do diretório para salvar as imagens segmentadas.")
    parser.add_argument("--threshold", type=float, default=0, help="Limiar para detectar vegetação (padrão: 0).")

    args = parser.parse_args()
    process_directory(args.input, args.output, args.threshold)