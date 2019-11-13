import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


MW_FREQ = 512
MAX_LENGTH = 2
MAX_SIZE = MAX_LENGTH * MW_FREQ


def parse_entry(data_entry):
    entry_id, entry_event, device, channel, code, size, signal = data_entry.rstrip().split('\t')
    entry_id, entry_event, code, size = int(entry_id), int(entry_event), int(code), int(size)
    signal = np.float32(signal.split(','))
    return entry_id, entry_event, device, channel, code, size, signal


def parse_data_file(filepath):
    print('\nParsing raw .txt data file ...')
    with open(filepath, 'r') as f:
        raw_data = f.readlines()

    meta_data, data = [], []
    for i in tqdm(range(len(raw_data))):
        data_entry = raw_data[i]
        entry_id, entry_event, device, channel, code, size, signal = parse_entry(data_entry)
        if code < 0:
            continue

        meta_data.append([entry_id, entry_event, device, channel, code, size])
        data.append(np.pad(signal, pad_width=(0, MAX_SIZE - len(signal))))
    print('Done!')
    return meta_data, np.array(data)


def main():
    parser = argparse.ArgumentParser(description='parse raw .txt data file')
    parser.add_argument('--data-dir', type=str, default='/home/filippo/datasets/mindwave/')
    parser.add_argument('--test-id', type=int, default=123)
    args = parser.parse_args()

    fp = os.path.join(args.data_dir, 'MW.txt')
    meta_data, data = parse_data_file(fp)
    print('Dataset shape: {}'.format(data.shape))

    sns.set()
    sns.lineplot(x=[i / MW_FREQ for i in range(len(data[args.test_id]))], y=data[args.test_id])
    plt.title('EEG signal {} corresponding to digit {}'.format(meta_data[args.test_id][0], meta_data[args.test_id][4]))
    plt.xlabel('Time (s)')
    plt.ylabel('EEG signal amplitude')
    plt.show()


if __name__ == '__main__':
    main()
