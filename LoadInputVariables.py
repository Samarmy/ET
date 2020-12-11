import socket
from datetime import datetime
import datetime
import pygeohash
from skimage import io
from skimage.transform import resize
import ee
import numpy as np
import stippy
import torch
from time import strptime, mktime
import ee
import pprint

service_account = 'fineet-paahuni@appspot.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, '../satelliteimagery-275021-d84dc86097ba.json')
# ee.Authenticate()
# ee.Initialize(credentials)
import calcs


class generate_input_data():
    def __init__(self, pix_cover=0.1, cloud_cov=0.1, geoh=None,
                 dataset=None, startT=None, endT=None,
                 impute_high_resolution=False, albums='colorado', is_output_image=True):

        self.startT = startT
        self.endT = endT
        self.pix_cover = pix_cover
        self.cloud_cov = cloud_cov
        self.geoh = geoh
        self.host_addr = socket.gethostbyname(socket.gethostname()) + ':15606'
        self.dataset = dataset
        self.albums = albums
        self.impute_high_resolution = impute_high_resolution
        self.target_image_size = (10, 10)
        self.is_output_image = is_output_image  # point or image
        self.platform = ''
        if self.dataset == 'modis-weekly':
            self.platform = 'MOD11A2'
        elif self.dataset == 'landsat':
            self.platform = 'Landsat8C1L1'
        elif self.dataset == 'viirs':
            self.platform = 'VNP21v001'
        else:
            self.platform = 'MOD11A1'

        if geoh is None:
            stip_iter = stippy.list_node_images(self.host_addr, platform=self.platform, album='colorado',
                                                min_pixel_coverage=1.0, source='raw', max_cloud_coverage=self.cloud_cov,
                                                start_timestamp=self.convertDateToEpoch(startT),
                                                end_timestamp=self.convertDateToEpoch(endT), recurse=True
                                                )
        else:
            stip_iter = stippy.list_node_images(self.host_addr, platform=self.platform, album='colorado',
                                                min_pixel_coverage=1.0, source='raw', max_cloud_coverage=self.cloud_cov,
                                                start_timestamp=self.convertDateToEpoch(startT),
                                                end_timestamp=self.convertDateToEpoch(endT), geocode=geoh
                                                )
        fileEndingIndex = {'MOD11A1': 1, 'MOD11A2': 1, 'VNP21v001': 2, 'Landsat8C1L1': 3}
        self.thermalBandIndex = {'MOD11A1': [1, 3], 'MOD11A2': [2, 5], 'VNP21v001': [2, 5], 'Landsat8C1L1': [3, 1]}
        self.datafilenames, self.timestamp = [], []
        for i in stip_iter:
            paths = []
            for file in i[1].files:
                if not file.path.endswith("-" + str(fileEndingIndex.get(self.platform)) + ".tif"):
                    continue
                paths.append(file.path)

            self.datafilenames.append(paths)
            self.timestamp.append(datetime.utcfromtimestamp(i[1].timestamp))
        ee.Initialize()

    def __len__(self):
        return len(self.datafilenames)

    def __getitem__(self, item):
        images = []
        if torch.is_tensor(item):
            item = item.tolist()
        for ind, filen in enumerate(self.datafilenames[item]):
            image = resize(io.imread(filen), self.target_image_size, anti_aliasing=True)
            for c in self.thermalBandIndex.get(self.platform):
                channel = torch.tensor(image[:, :, c].astype(np.float32))
                images.append(channel)

        all_thermal_bands = torch.stack(images).squeeze(0)
        return all_thermal_bands

    def get_ndvi(self, inp_image):
        # Update for different dataset
        return (inp_image[:, 2] - inp_image[:, 11]) / (inp_image[:, 2] + inp_image[:, 11])

    def predict_missing_band(self):
        '''
        :return: ML Model generated high spatial and temporal resolution temperature bands
        '''
        return

    def temp_bands(self):
        '''
        :return: Return tmax, tmin using any one of the dataset-
        0 - Gridmet
        1 - Landsat 8
        2 - MODIS Daily
        3 - MODIS 8-day
        4 - VIIRS
        '''
        return

    def convertDateToEpoch(self, inputDate):
        dateF = strptime(inputDate + ' 00:00', '%Y-%m-%d %H:%M')
        epochTime = mktime(dateF)
        return int(epochTime)

    def convertEpochToDate(self, inputEpoch):
        return datetime.datetime.fromtimestamp(inputEpoch).strftime('%Y-%m-%d')

    def getGeoH(self, lat, long, length=3):
        return pygeohash.encode(lat, long, precision=length)

    def input_features(self, isDatasetDownloaded=False):
        '''
        :return: tmax, tmin, ea, rs, uz, zw, elev, lat
        '''
        if not isDatasetDownloaded:
            images = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').filterDate(self.startT, self.endT)
            # print("Found {} images for GRIMET data in given time frame".format(images.size()))
            # tmax, tmin = self.temp_bands()
            for img in images:
                tmin = img.select(['tmmn']).subtract(273.15).rename(['tmin'])
                tmax = img.select(['tmmx']).subtract(273.15).rename(['tmax'])
                uz = img.select(['vs']).rename(['uz'])
                rs = img.select(['srad']).multiply(0.0864).rename(['rs'])
                ea = calcs._actual_vapor_pressure(
                    q=img.select(['sph']),
                    pair=calcs._air_pressure(elev, 'asce')).rename(['ea'])
                yield tmin, tmax, uz, rs, ea

        # Currently supports Gridmet dataset using EE tool;
        else:
            if self.geoh is None:
                stip_iter = stippy.list_node_images(self.host_addr, platform='GRIDMET', album='colorado',
                                                    min_pixel_coverage=1.0, source='raw',
                                                    max_cloud_coverage=self.cloud_cov,
                                                    start_timestamp=self.startT, end_timestamp=self.endT, recurse=True
                                                    )
            else:
                stip_iter = stippy.list_node_images(self.host_addr, platform='GRIDMET', album='colorado',
                                                    min_pixel_coverage=1.0, source='raw',
                                                    max_cloud_coverage=self.cloud_cov,
                                                    start_timestamp=self.startT, end_timestamp=self.endT,
                                                    geocode=self.geoh
                                                    )

        return


if __name__ == '__main__':
    loaded_data = generate_input_data(startT='2018-03-01', endT='2018-08-01')
    loaded_data.input_features()
