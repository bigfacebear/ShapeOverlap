#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from six.moves import urllib as smurllib
import zipfile
import tarfile

def maybe_download_and_extract(data_dir, data_url):
    """Downloads and extracts the zip from electronneutrino, if necessary"""
    if os.path.exists(data_dir):
        return

    download_directory = './download'
    if not os.path.exists(download_directory):
        os.makedirs(download_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(download_directory, filename)
    if not os.path.exists(filepath):
        print_progress_bar(0, 100,
                           prefix='Downloading ' + filename + ":", suffix='Complete', length=50,
                           fill='█')


        def _progress(count, block_size, total_size):
            print_progress_bar(float(count * block_size) / float(total_size) * 100.0, 100,
                               prefix='Downloading ' + filename + ":", suffix='Complete', length=50,
                               fill='█')


        filepath, _ = smurllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    extracted_dir_path = reduce(lambda p1, p2:os.path.join(p1, p2), data_dir.split('/')[:-1])
    if not os.path.exists(extracted_dir_path):
        os.makedirs(extracted_dir_path)

    print('Extracting files...', end=' ')
    # zip_ref = zipfile.ZipFile(filepath, 'r')
    # zip_ref.extractall(extracted_dir_path)
    # zip_ref.close()
    tar = tarfile.open(filepath)
    tar.extractall(extracted_dir_path)
    tar.close()
    print('done')


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill="█"):
    """
    Call in a loop to create terminal progress bar. Based on https://stackoverflow.com/a/34325723
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix) + '\r')
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()
        print()
        sys.stdout.write("")


if __name__ == '__main__':
    maybe_download_and_extract('./Dataset', 'https://pandownload.zju.edu.cn/download/2f822fe6c8ae4610b619cf6b7ba99529/0f4b9bde42b6c9cb9597ac5cb8052f1dbff6e64e31efc8e9d33db7bd22be23a7/Dataset.zip')