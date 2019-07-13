import os
from pathlib import Path
from random import seed, shuffle


seed(666)


def split_dataset(dataset_path, train_valid_ratio = 0.2):
    p_folder = Path(dataset_path)
    p_midi_list = list(p_folder.glob('*.mid'))
    shuffle(p_midi_list)

    p_train = p_folder / 'train'
    p_valid = p_folder / 'valid'
    if not(p_train.exists()):
        p_train.mkdir()
    if not(p_valid.exists()):
        p_valid.mkdir()

    data_num = len(p_midi_list)
    train_num = int(data_num * (1 - train_valid_ratio))
    for i in range(train_num):
        file_name = p_midi_list[i].name.split('.')[0]
        file_path = p_folder / (file_name + '.*')
        cmd = 'mv ' + str(file_path) + ' ' + str(p_train)
        os.system(cmd)
        print(cmd)

    for i in range(train_num, data_num):
        file_name = p_midi_list[i].name.split('.')[0]
        file_path = p_folder / (file_name + '.*')
        cmd = 'mv ' + str(file_path) + ' ' + str(p_valid)
        os.system(cmd)
        print(cmd)

if __name__ == '__main__':
    dataset_path = 'dataset_tmp' 
    split_dataset(dataset_path)
