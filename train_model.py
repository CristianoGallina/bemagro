import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

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

# Dataset para carregar imagens e máscaras
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Escala de cinza

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Normalizar máscara para valores 0 e 1
        mask = (mask > 0).float()

        return image, mask

# Função principal para treinamento
def main(args):
    # Definir transformações
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.ToTensor()

    # Criar dataset e dataloader
    dataset = SegmentationDataset(
        args.rgb,
        args.groundtruth,
        image_transform=image_transform,
        mask_transform=mask_transform
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Inicializar o modelo
    model = UNet()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Treinar o modelo
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(dataloader):
            # Enviar para GPU, se disponível
            images = images.cuda() if torch.cuda.is_available() else images
            masks = masks.cuda() if torch.cuda.is_available() else masks

            # Zero gradiente
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calcular a perda
            loss = criterion(outputs.squeeze(1), masks.squeeze(1))

            # Backward pass e otimização
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

    # Salvar o modelo treinado
    torch.save(model.state_dict(), args.modelpath)
    print(f"Modelo salvo em {args.modelpath}")

# Argumentos da linha de comando
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar modelo de segmentação de vegetação.")
    parser.add_argument("--rgb", required=True, help="Caminho do diretório com as imagens RGB em blocos.")
    parser.add_argument("--groundtruth", required=True, help="Caminho do diretório com as máscaras de vegetação em escala de cinza.")
    parser.add_argument("--modelpath", required=True, help="Caminho para salvar o modelo treinado.")
    parser.add_argument("--epochs", type=int, default=15, help="Número de épocas para treinamento.")

    args = parser.parse_args()
    main(args)

