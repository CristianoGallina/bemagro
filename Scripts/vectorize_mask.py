import sys
from osgeo import gdal, ogr

def vectorize_mask(mask_path, output_path):
    # Abrir a imagem raster
    raster = gdal.Open(mask_path)
    
    # Criar um driver para o formato de saída (GeoJSON)
    driver = ogr.GetDriverByName("GeoJSON")
    if driver is None:
        raise ValueError("O driver GeoJSON não está disponível.")

    # Criar a camada de vetor (GeoJSON)
    out_datasource = driver.CreateDataSource(output_path)
    if out_datasource is None:
        raise ValueError(f"Não foi possível criar o arquivo: {output_path}")
    
    # Definir o tipo de camada de vetor (polígono)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # Definindo o CRS para WGS 84 (EPSG:4326)

    # Criar a camada no arquivo GeoJSON
    out_layer = out_datasource.CreateLayer("layer", srs, ogr.wkbPolygon)

    # Adicionar um campo (atributo) para armazenar os valores do raster
    field = ogr.FieldDefn("value", ogr.OFTInteger)
    out_layer.CreateField(field)

    # Vetorização usando o GDAL
    gdal.Polygonize(raster.GetRasterBand(1), None, out_layer, 0, [], callback=None)

    print(f"Arquivo GeoJSON salvo em: {output_path}")

# Caminhos da máscara e do arquivo de saída
mask_path = sys.argv[sys.argv.index("--mask") + 1]
output_path = sys.argv[sys.argv.index("--output") + 1]

vectorize_mask(mask_path, output_path)