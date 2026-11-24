import os
import glob
import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

DIRETORIO_KAGGLE = "/kaggle/input/datasets/yoctoman/shcxr-lung-mask/mask/*.png"
ARQUIVO_SAIDA = "/kaggle/working/estatisticas_completas_256.csv"
WORKERS = os.cpu_count()


def calcular_todas_metricas(caminho_imagem):
    nome_arquivo = os.path.basename(caminho_imagem)

    img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {'arquivo': nome_arquivo, 'erro': 'Falha ao ler imagem'}

    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)

    mascara_binaria = (img > 127).astype(np.uint8)
    tamanho_total_pixels = np.sum(mascara_binaria)

    resultado = {
        'arquivo': nome_arquivo[7:13],
        'preenchimento_total_pct': tamanho_total_pixels / (256 * 256)
    }

    if tamanho_total_pixels == 0:
        resultado.update({
            'tamanho_pixels_total': 0, 'aspect_ratio_total': 0, 'solidity_total': 0,
            'eccentricity_total': 0, 'perimetro_total': 0, 'diametro_equivalente_total': 0,
            'esq_tamanho_pixels': 0, 'esq_aspect_ratio': 0, 'esq_solidity': 0,
            'esq_eccentricity': 0, 'esq_perimetro': 0, 'esq_diametro_eq': 0,
            'dir_tamanho_pixels': 0, 'dir_aspect_ratio': 0, 'dir_solidity': 0,
            'dir_eccentricity': 0, 'dir_perimetro': 0, 'dir_diametro_eq': 0,
            'erro': 'Mascara totalmente vazia'
        })
        return resultado

    prop_global = regionprops(mascara_binaria)[0]
    minr, minc, maxr, maxc = prop_global.bbox
    altura = maxr - minr
    largura = maxc - minc

    resultado.update({
        'tamanho_pixels_total': tamanho_total_pixels,
        'aspect_ratio_total': largura / altura if altura > 0 else 0,
        'solidity_total': prop_global.solidity,
        'eccentricity_total': prop_global.eccentricity,
        'perimetro_total': prop_global.perimeter,
        'diametro_equivalente_total': prop_global.equivalent_diameter
    })

    mascara_rotulada = label(mascara_binaria, background=0)
    propriedades = regionprops(mascara_rotulada)

    propriedades = sorted(propriedades, key=lambda x: x.area, reverse=True)[:2]
    propriedades = sorted(propriedades, key=lambda p: p.centroid[1])

    prefixos = ['esq_', 'dir_']

    for idx, prop in enumerate(propriedades):
        prefixo = prefixos[idx]

        minr, minc, maxr, maxc = prop.bbox
        altura = maxr - minr
        largura = maxc - minc

        resultado.update({
            f'{prefixo}tamanho_pixels': prop.area,
            f'{prefixo}aspect_ratio': largura / altura if altura > 0 else 0,
            f'{prefixo}solidity': prop.solidity,
            f'{prefixo}eccentricity': prop.eccentricity,
            f'{prefixo}perimetro': prop.perimeter,
            f'{prefixo}diametro_eq': prop.equivalent_diameter
        })

    if len(propriedades) == 1:
        prefixo = prefixos[1]
        resultado.update({
            f'{prefixo}tamanho_pixels': 0, f'{prefixo}aspect_ratio': 0,
            f'{prefixo}solidity': 0, f'{prefixo}eccentricity': 0,
            f'{prefixo}perimetro': 0, f'{prefixo}diametro_eq': 0
        })

    return resultado


lista_imagens = sorted(glob.glob(DIRETORIO_KAGGLE))
resultados = []

with ProcessPoolExecutor(max_workers=WORKERS) as executor:
    resultados_iter = executor.map(calcular_todas_metricas, lista_imagens)

    for resultado in tqdm(resultados_iter, total=len(lista_imagens)):
        if 'erro' not in resultado or resultado.get('erro') == 'Mascara totalmente vazia':
            resultado.pop('erro', None)
            resultados.append(resultado)

df = pd.DataFrame(resultados)
df.to_csv(ARQUIVO_SAIDA, index=False)