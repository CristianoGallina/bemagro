import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import tifffile as tiff  # Usado para salvar imagens em formato .tif

# Definir arquitetura do modelo (U-Net simplificada)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Camadas do modelo U-Net (simplificadas)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Função para carregar e segmentar a imagem
def run_inference(model, rgb_image_path, output_path):
    # Carregar a imagem RGB
    image = Image.open(rgb_image_path).convert("RGB")
    
    # Transformação da imagem
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)  # Adiciona dimensão de batch
    
    # Enviar para GPU, se disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image_tensor = image_tensor.to(device)
    
    # Inferência
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    
    # Converter a saída para uma máscara binária
    output_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    output_mask = output.squeeze().cpu().numpy()
    print(f"Valores de saída do modelo: {np.min(output_mask)} - {np.max(output_mask)}")

    # Salvar a máscara em um arquivo .tif
    tiff.imsave(output_path, output_mask)
    print(f"Máscara salva em {output_path}")

# Função principal de execução
def main():
    parser = argparse.ArgumentParser(description="Segmentação de vegetação em ortomosaicos.")
    parser.add_argument("--rgb", required=True, help="Caminho da imagem RGB a ser segmentada.")
    parser.add_argument("--modelpath", required=True, help="Caminho do modelo treinado.")
    parser.add_argument("--output", required=True, help="Caminho onde salvar a máscara segmentada.")
    args = parser.parse_args()

    # Carregar o modelo treinado
    model = UNet()
    model.load_state_dict(torch.load(args.modelpath, map_location=torch.device('cpu')))
    model.eval()

    # Executar a inferência
    run_inference(model, args.rgb, args.output)

if __name__ == "__main__":
    main()




