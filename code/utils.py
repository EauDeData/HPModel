import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streetview
import os
import multiprocessing as mp

def download_split(dataframe, p, n_process):
    base = '../panoramics/'
    n=1
    for num in range(p, len(dataframe), n_process):
        num = dataframe.index[num]
        coords = dataframe['coords'][num]
        if type(coords) != type('henlo'): continue
        url = dataframe['url'][num]
        x, y = coords.replace('(', '').replace(')', '').split()
        x = float(x)
        y = float(y)
        id_ = url.split('/')[-1]
        dir_ = base + id_
        try:
            os.mkdir(dir_)
        except FileExistsError:
            continue
        panoids = streetview.panoids(lat=x, lon=y)
        for i in range(1): #range(len(panoids)): TODO: We can change this and take the whole scenario
            newDir = dir_+'/'+str(i)
            try:
                os.mkdir(newDir)
            except FileExistsError:
                pass
            panoid = panoids[i]['panoid']
            meta = streetview.tiles_info(panoid)
            streetview.download_tiles(meta, newDir)
        n+=1
        print(f'Working on {num} out of {len(dataframe)}\t', end = '\r')

def download_tiles():
    dataframe = pd.read_csv('../data/data_harassment.tsv', delimiter = '\t', names = ['url', 'coords', 'categories', 'text'])
    dataframe = dataframe[dataframe.coords!='( )']

    n = 16
    process_batch = [mp.Process(target=download_split, args=(dataframe, p, n, )) for p in range(n)]
    for n, p in enumerate(process_batch):
        print('Process', n, 'started')
        p.start()
    for n, p in enumerate(process_batch):
        p.join()
        print('Process', n, 'finished') 


def get_full_image(directory):

    images = os.listdir(directory)
    rows = [int(x.split('x')[0].split('_')[-1]) for x in images]
    columns = [int(x.split('x')[-1].split('.')[0]) for x in images]

    max_i, max_j = max(columns)+1, max(rows)+1
    full_image = np.zeros((max_i, max_j)).tolist()

    for image in images:

        i, j = int(image.split('x')[-1].split('.')[0]), int(image.split('x')[0].split('_')[-1])
        full_image[i][j] = cv2.imread(directory+image, cv2.IMREAD_COLOR)

    return np.vstack([np.hstack(x) for x in full_image]) #OMG this function is SICK

if __name__ == "__main__":
    plt.imshow(get_full_image('/home/adri/Desktop/projects/harassment/panoramics/3decec6dfa/1/'))
    plt.show()