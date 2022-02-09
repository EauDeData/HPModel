import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streetview
from googletrans import Translator, constants
import os
import multiprocessing as mp
from bs4 import BeautifulSoup
import requests as r
import re


def scrap_tables():

    all_links = []

    for i in range (1, 185):
        soup = BeautifulSoup(r.get(f'https://harassmap.org/en/reports?page={i}').text)
        links = [[y for y in str(x).split('\n') if 'data-href=' in y][0] for x in soup.find_all('', class_ = 'row-link')]
        all_links.extend(links)
        print(len(all_links), end = '\r')

    clean_links = []
    for link in all_links:
        match = re.search(r'href=[\'"]?([^\'" >]+)', link)
        clean_links.append(match.group(1))
    open('links.txt', 'w').write('\n'.join(clean_links))


    rows = []
    for n, url in enumerate(clean_links):
        print(n, url, '\t', end = '\r')
        page = BeautifulSoup(r.get(url).text)
        caption = page.find('', class_ = 'read-more')
        if not caption: caption = None
        else: caption = caption.text.strip()
        coords = page.find('', class_ = 'latlng')
        if not coords: coords = None
        else: coords= coords.text.strip()
        categories = '|'.join([i.text.strip() for i in page.find_all('', class_ = 'category mb-1')] if i else None)
        rows.append([url, coords, categories, caption])
    open('../data/data_harassment.tsv', 'w').write('\n'.join(['\t'.join(row) for row in rows]))
    
    return 1

def parse_ftb(file):
    dataframe = pd.read_csv(file, na_values='nan', quotechar='"')
    columns = ["the_geom", "description_negative", "description_positive", "description_translation", "gender", "location_latlng_lat", "location_latlng_lng", "spot_type"]
    dataframe = dataframe[columns]
    # Select only cases with a description
    dataframe = dataframe[~((dataframe.description_negative.isna()) & (dataframe.description_positive.isna()))]
    # Select only cases where coords are known
    dataframe = dataframe[~((dataframe.location_latlng_lat.isna()) & (dataframe.location_latlng_lng.isna()))]
    # Create coord tuple (lat, lon)
    dataframe["coords"] = list(zip(dataframe.location_latlng_lat, dataframe.location_latlng_lng))
    dataframe = dataframe.drop(["location_latlng_lat", "location_latlng_lng"], axis=1)
    # Create target {"good": 1, "bad": 0}
    encode = {"good": 1, "bad": 0}
    dataframe["target"] = dataframe.spot_type.replace(encode)
    dataframe = dataframe.drop(["spot_type"], axis=1)
    # Create description: translate and remove non-text
    #trans_pos = dataframe['description_positive'].dropna()[:10].apply(lambda x: translator.translate(x, dest='en').text)    
    
    return dataframe

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
    #plt.imshow(get_full_image('/home/adri/Desktop/projects/harassment/panoramics/3decec6dfa/1/'))
    #plt.show()
    dataframe = parse_ftb("/home/gerard/Desktop/AI4SH/HPModel/data/csv-ftb/ftb_delhi_archive.csv")
