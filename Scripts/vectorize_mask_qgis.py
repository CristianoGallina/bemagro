import sys
import os
# Defina os caminhos
os.environ['PATH'] += r";C:/Program Files/QGIS 3.40.1/bin"
os.environ['PATH'] += r";C:/Program Files/QGIS 3.40.1/apps/qgis/bin"
sys.path.append('C:/Program Files/QGIS 3.40.1/apps/qgis/python')
sys.path.append("C:/Program Files/QGIS 3.40.1/apps/qgis/plugins")

from qgis.core import QgsApplication
from qgis import processing



# Inicializa a aplicação QGIS sem interface gráfica
qgis_path = "C:/Program Files/QGIS 3.40.1/apps/qgis"
QgsApplication.setPrefixPath(qgis_path, True)
qgs = QgsApplication([], False)
qgs.initQgis()

# Carregar o plugin Processing diretamente
from qgis import processing

# Função para vetorização
def vectorize_mask(mask_path, output_path):
    # Parâmetros de vetorização
    params = {
        'INPUT': mask_path,
        'BAND': 1,
        'FIELD_NAME': 'value',
        'EIGHT_CONNECTEDNESS': False,
        'EXTRA': '',
        'OUTPUT': output_path,
        'SRC_METHOD': 'NO_GEOTRANSFORM'  # Corrige o erro de transformação
    }

    # Executar o algoritmo de vetorização
    processing.run("gdal:polygonize", params)
    print(f"Arquivo GeoJSON salvo em: {output_path}")

# Caminhos da máscara e do arquivo de saída
mask_path = sys.argv[sys.argv.index("--mask") + 1]
output_path = sys.argv[sys.argv.index("--output") + 1]

try:
    vectorize_mask(mask_path, output_path)
finally:
    # Finalizar o QGIS
    qgs.exitQgis()

