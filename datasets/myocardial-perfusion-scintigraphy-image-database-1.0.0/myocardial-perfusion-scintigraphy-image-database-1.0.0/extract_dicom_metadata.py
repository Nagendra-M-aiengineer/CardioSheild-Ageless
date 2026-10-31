import os
import pydicom
import pandas as pd

dicom_path = r"C:\Users\wpcal\Dropbox\Arquivos Pacheco 02_05_2022\Programacao\Ambiente_VirtualCS\Dicom\1.2.840.4267.32.102843376980518437893525476318362476257.dcm"

# Tenta carregar o arquivo e extrair todos os metadados
try:
    ds = pydicom.dcmread(dicom_path)
    metadata = []

    print("=== Metadados disponíveis ===")
    for elem in ds.iterall():
        tag = str(elem.tag)
        name = elem.keyword
        value = str(elem.value)
        metadata.append({"Tag": tag, "Keyword": name, "Value": value})
        print(f"{tag} | {name} | {value}")

    # Salva todos os metadados em CSV
    df = pd.DataFrame(metadata)
    df.to_csv("demographics_full.csv", index=False)
    print("\nTodos os metadados exportados para demographics_full.csv com sucesso.")

except Exception as e:
    print(f"Erro ao ler o arquivo DICOM: {e}")
