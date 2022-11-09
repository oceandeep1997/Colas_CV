import urllib.error
from urllib.request import urlretrieve
import warnings
import pickle
from pyunpack import Archive

import re
import os
import shutil
# To ensure backwards compatibility we use the List and Dict classes
from typing import List, Dict

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By

from tqdm import tqdm
from tqdm_jupyter import tqdm_jupyter

import pandas as pd
import geopandas as gpd

warnings.simplefilter(action='ignore', category=FutureWarning)


class ScrapeCoordinates:
    def __init__(self):
        self.urls_to_download = None

    def get_download_urls(self) -> List[str]:
        """This method creates a list with the download links to the coordinate files."""
        # Setting the option to open the Firefox driver in headless mode (you won't see a browser window)
        options = Options()
        options.headless = True
        # Creating the driver
        driver = webdriver.Firefox(service=Service(executable_path='./geckodriver'), options=options)
        driver.get('https://geoservices.ign.fr/bdcarto')
        # Navigating to the html part that contains the urls
        download_section = driver.find_element(By.XPATH,
                                               '/html/body/div[2]/div[1]/div/section/div[3]/section/div/article/div[2]/'
                                               'div/div[1]/div/div[2]/div[3]/div/div/div[2]')
        # Retrieving the download links
        self.urls_to_download = download_section.find_elements(By.CSS_SELECTOR, 'a')
        # We only want to keep the links for the departments
        limit = download_section.find_elements(By.CSS_SELECTOR, 'h3')[1].location['y']
        self.urls_to_download = [url.get_attribute('href') for url in self.urls_to_download
                                 if url.location['y'] < limit]
        # Closing the driver again
        driver.close()
        return self.urls_to_download

    def download_files(self, urls_to_download: List[str] = None, coordinates_folder: str = 'Coordinates',
                       files_to_retrieve: List[str] = None) -> None:
        """This method will download the compressed files from the provided urls, unpack them, look for the airport
        coordinate files, store them in a different place and then discard the other downloaded files.
        Args:
            urls_to_download: urls to retrieve the compressed files from
            coordinates_folder: folder you want to store the files with the coordinates to
            files_to_retrieve: name(s) of files to retrieve from the downloaded files"""

        # Handling the input
        # Creating the folder to store the files with the coordinates in
        if coordinates_folder not in os.listdir():
            os.mkdir(coordinates_folder)
        if urls_to_download is None:
            urls_to_download = self.urls_to_download
        if files_to_retrieve is None:
            files_to_retrieve = ['AERODROME.cpg',
                                 'AERODROME.dbf',
                                 'AERODROME.prj',
                                 'AERODROME.shp',
                                 'AERODROME.shx',
                                 'PISTE_D_AERODROME.cpg',
                                 'PISTE_D_AERODROME.dbf',
                                 'PISTE_D_AERODROME.prj',
                                 'PISTE_D_AERODROME.shp',
                                 'PISTE_D_AERODROME.shx']

        # Creating the folder to unpack the downloaded zip files in
        if 'Unpacking' not in os.listdir():
            os.mkdir('Unpacking')

        # Create list for urls that were not downloadable
        url_errors = []
        # Looping over the download links
        for url in tqdm(urls_to_download, desc='Downloading...: '):
            # Downloading the compressed file
            try:
                file_name, headers = urlretrieve(url, filename='Unpacking/zipped.7z')
            except urllib.error.URLError:
                url_errors.append(url)
                continue
            # Unpacking the compressed file to the folder "Unpacking"
            Archive(filename=file_name).extractall('Unpacking')
            # Looping over all the files in the folder "Unpacking"
            for path, folders, file_names in os.walk('Unpacking'):
                # If there is an "AERODROME.shp" file in one of the folders
                if set(files_to_retrieve).intersection(file_names):
                    for file in set(files_to_retrieve).intersection(file_names):
                        # Copy it to the folder with all the coordinate files
                        shutil.copy(src=f'{path}/' + file,
                                    dst=f'{coordinates_folder}/' + re.search('_(D\d+)', path).groups()[0] + '_' + file)
                    # Remove the "Unpacking" folder
                    shutil.rmtree(path='Unpacking')
                    # Recreate an empty "Unpacking" folder
                    os.mkdir('Unpacking')
                    # Since there is assumed to be only one file per region with the airport coordinates we can break
                    # the loop
                    break
        if url_errors:
            url_errors_str = ",\n".join(url_errors)
            print(f'Could not download: {url_errors_str}')
        shutil.rmtree(path='Unpacking')

    @staticmethod
    def create_concatenated_geodf(path: str, regex: str = 'D\d+_A.+\.shp', crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
        """This method creates one DataFrame containing geo information on all airports.
        Args:
            path: path to the files you want to concatenate
            regex: regex to filter for the files you want to concatenate
            crs: coordinate reference code to transform the coordinates to, since only files with the same reference
                 system can be concatenated. The most common reference system is the World Geodetic System 1984 which
                 has the code 'EPSG:4326'"""
        geodfs = [gpd.read_file(f'{path}/{file}') for file in os.listdir(path=path) if re.findall(regex, file)]
        geodfs_crs = [geodf.to_crs(crs=crs) for geodf in geodfs]
        # The GeoDataFrame wrapping is only necessary for type checkers
        return gpd.GeoDataFrame(pd.concat(geodfs_crs, ignore_index=True))


class ScrapePictures:
    def __init__(self):
        self.relevant_urls = None

    def get_download_urls(self) -> List[str]:
        driver = webdriver.Firefox(service=Service(executable_path='./geckodriver'))
        driver.get('https://geoservices.ign.fr/bdortho')
        download_section = driver.find_element(By.XPATH,
                                               '/html/body/div[2]/div[1]/div/section/div[3]/section/div/article/'
                                               'div[2]/div/div[1]/div/div[2]/div[3]/div/div/div[2]')
        all_url_elements = download_section.find_elements(By.CSS_SELECTOR, 'a')
        # Getting the location until which we actually want the url since the page also contains old images which we are
        # not interested in
        limit = download_section.find_elements(By.CSS_SELECTOR, 'h3')[1].location['y']
        self.relevant_urls = [url.text for url in all_url_elements if url.location['y'] < limit]
        driver.close()
        return self.relevant_urls

    @staticmethod
    def download_files(urls: List[str]) -> None:
        """This method downloads files from urls."""
        url_errors = []
        if 'Unpacking' not in os.listdir():
            os.mkdir('Unpacking')
        try:
            ide = get_ipython().__class__.__name__
        except NameError:
            ide = ''
        if not ide:
            pbar = tqdm(urls, leave=False)
        else:
            pbar = tqdm_jupyter(urls)
        for url in pbar:
            pbar.set_description(f'Downloading partial files ...: {re.search("[1-9]+$", url)}/{len(pbar)}')
            try:
                urlretrieve(url, f'Unpacking/{url.split("/")[-1]}')
            except urllib.error.URLError:
                url_errors.append(url)

        if url_errors:
            url_errors_str = ",\n".join(url_errors)
            print(f'The following files could not be downloaded: {url_errors_str}')

    @staticmethod
    def join_partial_zip_files(file_name: str, source_path: str = '.', dest_file: str = 'joined.7z') -> None:
        """This method joines multi-volume compressed files.
        Args:
            file_name: the names of all the files you want to join
            source_path: path where the partial files you want to join can be found
            dest_file: the name of the file with the joined content"""

        partial_files = [file for file in os.listdir(source_path) if file.startswith(file_name)]
        with open(dest_file, 'ab') as output_file:
            for partial_file in partial_files:
                with open(f'{source_path}/{partial_file}', 'rb') as input_file:
                    output_file.write(input_file.read())

    @staticmethod
    def compare_all_coordinates(df1: gpd.GeoDataFrame, df2: gpd.GeoDataFrame,
                                crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
        """This method compares the geometries in df1 and df2 and only returns the rows from df1 that intersect with
        df2.
        Args:
            crs: reference coordinate system to convert both GeoDataFrames to to allow to compare them"""
        # Transforming to the common crs
        df1, df2 = df1.to_crs(crs=crs), df2.to_crs(crs=crs)

        intersection = []
        for index, geometry in enumerate(df1['geometry']):
            if any(df2.intersects(geometry)):
                intersection.append(index)
        return df1.iloc[intersection, :]

    def create_url_dict(self, urls: List[str] = None, file_name: str = 'urls_dict.pkl') -> Dict[str, List[str]]:
        """This method creates a dictionary where the value for one key is a list of all the urls for a multi-volume
        compression file."""
        if urls is None:
            urls = self.relevant_urls
        files = dict()
        for url in urls:
            name = url.split('/')[-1]
            if name[-1].isdigit():
                key = re.findall('(.+).7z', name)[0]
                if key in files:
                    files[key] += [url]
                else:
                    files[key] = [url]
        with open('urls_dict.pkl', 'wb') as f:
            pickle.dump(files, f)
            print(f'Saved urls under: {file_name}')
        return files

    @staticmethod
    def query_image_api(coordinates: str, pixels: int = 1000, crs: str = 'EPSG:4326',
                        file_format: str = 'image/jpeg') -> None:
        query = f"https://wxs.ign.fr/essentiels/geoportail/r/wms?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&STYLES=&" \
                f"LAYERS=ORTHOIMAGERY.ORTHOPHOTOS&" \
                f"CRS={crs}&" \
                f"BBOX={coordinates}&" \
                f"WIDTH={pixels}&" \
                f"HEIGHT={pixels}&" \
                f"FORMAT={file_format}"
        file_name, headers = urlretrieve(query, f'{crs}_{coordinates}.')
        print(f'Image saved as: {file_name}')

    def extract_relevant_files(self, coordinates_path: str = 'Coordinates', images_path: str = 'Images',
                               urls_file: str = 'urls_dict.pkl') -> None:
        """This method brings it all together: it downloads the zip files, unpacks them and extracts the relevant
        images.
        Args:
            coordinates_path: path to the files containing the coordinates of the airports
            images_path: path where the relevant images should be extracted to
            urls_file: path to pickle file where the urls were stored"""
        if images_path not in os.listdir():
            os.mkdir(images_path)
        # Loading the dictionary with the urls to download the images from
        with open(urls_file, 'rb') as f:
            urls_dict = pickle.load(f)
        # Loading the airports geo-information
        df_airport_coordinates = ScrapeCoordinates.create_concatenated_geodf(path=coordinates_path)
        # List to capture image folders that did not contain a .shp file
        no_shp_file = []
        # Since jupyter behaves different from the terminal, we have to account for it here
        try:
            ide = get_ipython().__class__.__name__
        except NameError:
            ide = ''
        if not ide:
            pbar = tqdm(urls_dict)
        else:
            pbar = tqdm_jupyter(urls_dict)
        for joined_name in pbar:
            pbar.set_description(f'Downloading department: {joined_name}')
            self.download_files(urls=urls_dict[joined_name])
            self.join_partial_zip_files(file_name=joined_name, source_path='Unpacking', dest_file='Unpacking/joined.7z')
            Archive(filename='Unpacking/joined.7z').extractall('Unpacking')
            # This variable will capture the names of the images we would like to keep
            relevant_names = None
            # Extracting the names of the relevant files (without file extensions)
            for path, folders, files in os.walk('Unpacking'):
                if 'dalles.shp' in files:
                    df_ortho_all = gpd.read_file(filename=f'{path}/dalles.shp')
                    # A pandas series with the file names of the relevant .jp2 files
                    relevant_files = self.compare_all_coordinates(df1=df_ortho_all, df2=df_airport_coordinates)['NOM']
                    # Since the .tab files might come in handy later as well, we just want the name of the images
                    relevant_names = [re.search('/([\w-]+).', name).groups()[0] for name in relevant_files]
            if relevant_names is None:
                no_shp_file.append([name for name in os.listdir('Unpacking') if name.find('.') < 0][0])
                continue
            # Copying the relevant files to images_path
            for path, folders, files in os.walk('Unpacking'):
                matches = [file for file in files for name in relevant_names if file.startswith(name)]
                for match in matches:
                    shutil.copy(src=f'{path}/' + match,
                                dst=f'{images_path}/' + re.search('_(D\d+)', path).groups()[0] + '_' + match)
                if matches:
                    shutil.rmtree('Unpacking')
                    break
