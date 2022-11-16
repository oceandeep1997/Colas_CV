import re
import os
import shutil
import requests
import warnings
import py7zr
import tempfile
import pandas as pd
import geopandas as gpd
import chromedriver_autoinstaller

from io import BytesIO
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

chromedriver_autoinstaller.install()
warnings.simplefilter(
    action='ignore',
    category=FutureWarning)


class ScrapeCoordinates:
    def __init__(self):
        self.urls_to_download = None


    def get_download_urls(
        self
    ) -> list[str]:
        """
        Create list with download links to coordinate files
        """
        options = Options()
        options.headless = True
        driver = webdriver.Chrome(options=options)
        driver.get('https://geoservices.ign.fr/bdcarto')
        # Navigating to the html part that contains the urls
        download_section = driver.find_element(
            By.XPATH,
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


    def download_files(
        self,
        coordinates_folder: str = 'Coordinates',
        files_to_retrieve: list[str] = [
            'AERODROME.cpg',
            'AERODROME.dbf',
            'AERODROME.prj',
            'AERODROME.shp',
            'AERODROME.shx',
            'PISTE_D_AERODROME.cpg',
            'PISTE_D_AERODROME.dbf',
            'PISTE_D_AERODROME.prj',
            'PISTE_D_AERODROME.shp',
            'PISTE_D_AERODROME.shx'
            ]
    ) -> None:
        """
        Download coordinate archives and extract pertinent files
        
        Parameters :
        -----
        - coordinates_folder: output directory for coordinates files
        - files_to_retrieve: files to keep from retrieved archives
        """

        # create folder to store coord files
        if coordinates_folder not in os.listdir():
            os.mkdir(coordinates_folder)

        urls_to_download = self.urls_to_download
        url_errors = []

        # loop over download links
        for url in tqdm(urls_to_download, desc='Downloading...: '):
            # try getting files
            try:
                # get data
                response = BytesIO(requests.get(url).content)
                # open data archive
                with py7zr.SevenZipFile(response, 'r') as archive:
                    # create temp dir
                    with tempfile.TemporaryDirectory() as tmp:
                        # extract archive in temp dir
                        archive.extractall(path=tmp)
                        # search temp dir for files to keep
                        for path, _, file_names in os.walk(tmp):
                            for file in set(files_to_retrieve).intersection(file_names):
                                shutil.copy(
                                    src=f'{path}/' + file,
                                    dst=f'{coordinates_folder}/' + re.search('_(D\d+)', path).groups()[0] + '_' + file)
            
            # store url if failed
            except:
                url_errors.append(url)           

        if url_errors:
            print('Could not download :')
            for url in url_errors: print(url)


    @staticmethod
    def create_concatenated_geodf(
        path: str,
        regex: str = 'D\d+_A.+\.shp',
        crs: str = 'EPSG:4326'
    ) -> gpd.GeoDataFrame:
        """
        Create DataFrame containing geo information on airports
        
        Parameters :
        ------------
        - path: path to files to concatenate
        - regex: regex to filter for files to concatenate
        - crs: coordinate reference code to transform coordinates to
        """
        geodfs = [gpd.read_file(f'{path}/{file}') for file in os.listdir(path=path) if re.findall(regex, file)]
        geodfs_crs = [geodf.to_crs(crs=crs) for geodf in geodfs]
        # The GeoDataFrame wrapping is only necessary for type checkers
        return gpd.GeoDataFrame(pd.concat(geodfs_crs, ignore_index=True))



if __name__ == "__main__":
    downloader = ScrapeCoordinates()
    downloader.get_download_urls()
    downloader.download_files(coordinates_folder='coords')