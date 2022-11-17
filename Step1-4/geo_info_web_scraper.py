import urllib.error

import re
import os
import shutil
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from urllib.request import urlretrieve
from pyunpack import Archive
from tqdm import tqdm


class ScrapeCoordinates:
    def __init__(self):
        self.driver = webdriver.Firefox(service=Service(executable_path='./geckodriver'))
        self.driver.get('https://geoservices.ign.fr/bdcarto')
        self.urls_to_download = None

    def get_download_urls(self):
        """This method creates a list with the download links to the coordinate files."""
        # Navigating to the html part that contains the urls
        download_section = self.driver.find_element(By.XPATH,
                                                    '/html/body/div[2]/div[1]/div/section/div[3]/section/div/article/'
                                                    'div[2]/div/div[1]/div/div[2]/div[3]/div/div/div[2]')
        # Retrieving the download links
        self.urls_to_download = download_section.find_elements(By.CSS_SELECTOR, 'a')
        self.urls_to_download = [url.text for url in self.urls_to_download]
        return self.urls_to_download

    def download_files(self, urls_to_download: list = None, coordinates_folder: str = 'Coordinates',
                       files_to_retrieve: list = None) -> None:
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
                                 'PISTE_AERODROME.cpg',
                                 'PISTE_AERODROME.dbf',
                                 'PISTE_AERODROME.prj',
                                 'PISTE_AERODROME.shp',
                                 'PISTE_AERODROME.shx']

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
                                    dst=f'{coordinates_folder}/' + re.search('_(R\d+_)', path).groups()[
                                        0] + file)
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


c_loader = ScrapeCoordinates()
c_loader.get_download_urls()
c_loader.download_files()
c_loader.driver.close()