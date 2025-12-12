import openslide
import pandas as pd
from pathlib import Path
import shutil
import csv


if __name__ == "__main__":
    file_path = '/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/PCa-EPE.xlsx'
    csv_file_path = '/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/PCa-EPE.csv'
    folder_path = '/data1/xiamy/PCa-EPE/WSI/'
    out_path = '/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/svs/'

    # 读取表名称
    df = pd.read_excel(file_path, engine='openpyxl')
    first_column_data = df.iloc[:, 0].tolist()
    second_column_data = df.iloc[:, 1].tolist()

    first_column = ['case_id']
    second_column = ['slide_id']
    third_column = ['label']
    for file in Path(folder_path).rglob('*'):
        if file.is_file():
            tempAllName = file.name
            temp = tempAllName.rsplit('.', 1)[0]
            for i in range(len(first_column_data)):
                curName = first_column_data[i]
                if curName == temp[0:len(temp) - 1]:
                    first_column.append(temp)
                    second_column.append(temp)
                    if second_column_data[i] == 0:
                        third_column.append('C1')
                    else:
                        third_column.append('C2')
                    break
    csvdata = list(zip(first_column, second_column, third_column))
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(csvdata)

    print("finished!")
