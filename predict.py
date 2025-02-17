# import osgeo
import os
import geopandas as gp
import pandas as pd
import rasterio
import numpy as np

from pathlib import Path
from skimage import io
from rasterio.crs import CRS

from base import Base
from matplotlib import pyplot as plt
from deep_learning_models.metrics.polis import Polis
from deep_learning_models.models.unet import Unet
from sklearn.metrics import precision_recall_fscore_support
from shapely.geometry import Point, Polygon
from shapely.ops import polygonize
from rasterio.plot import reshape_as_image


def _visualize_results(**images):
    plt.figure(figsize=(16, 5))
    for i, (name, image_) in enumerate(images.items()):
        plt.subplot(1, len(images), i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image_)
    plt.show()

class Predict(Base):
    def __init__(self, test_generator):
        super().__init__()
        self.test_model = Unet().load_model()
        self.predict_weights_file = test_generator.predict_weights_file
        self.test_model.load_weights(os.path.join(self.weights_dir, self.predict_weights_file))
        self.test_generator = test_generator

        self.prediction_dir = os.path.join(self.test_dir, self.patches_dir, 'prediction')
        self.mosaic_dir = os.path.join(self.test_dir, self.patches_dir, 'mosaic')
        self.vector_dir = os.path.join(self.test_dir, self.patches_dir, 'vector')
        self.watershed_dir = os.path.join(self.test_dir, self.patches_dir, 'watershed')
        self.watershed_geo_dir = os.path.join(self.test_dir, self.patches_dir, 'watershed_geo')
        self.watershed_expand_dir = os.path.join(self.test_dir, self.patches_dir, 'watershed_expand')
        self.polis_dir = os.path.join(self.test_dir, self.patches_dir, 'polis')

        self.output_dirs = [self.prediction_dir,
                            self.mosaic_dir,
                            self.vector_dir,
                            self.watershed_dir,
                            self.watershed_geo_dir,
                            self.watershed_expand_dir,
                            self.polis_dir]

        self._create_output_dirs()
        self.images_list = os.listdir(os.path.join(self.test_dir, self.images_dir))
        self.masks_list = os.listdir(os.path.join(self.test_dir, self.masks_dir))

    def _create_output_dirs(self):
        for output_folder in self.output_dirs:
            Path(output_folder).mkdir(parents=True, exist_ok=True)

    def create_predictions(self):
        for _ in range(1000000):
            try:
                image, mask = self.test_generator[_]
                for i in range(len(self.test_generator.get_name(_))):
                    filename = self.test_generator.get_name(_)[i]
                    extents = self.test_generator.get_extent(_)[i]
                    pr_gradient = self.test_model.predict(image)[i]
                    transform = rasterio.transform.from_bounds(*extents, width=self.image_size, height=self.image_size)
    
                    if self.visualize_predictions:
                        _visualize_results(
                            input_image=image.squeeze(),
                            input_mask=mask.squeeze(),
                            predicted_prob=pr_gradient.squeeze(),
                        )
    
                    if self.write_predictions:
                        # https://gis.stackexchange.com/questions/279953/numpy-array-to-gtiff-using-rasterio-without-source
                        # -raster
                        new_dataset = rasterio.open(os.path.join(self.prediction_dir, 'prediction' + '_' + filename),
                                                    'w',
                                                    driver='GTiff',
                                                    transform=transform,
                                                    height=self.image_size,
                                                    width=self.image_size,
                                                    count=1,
                                                    crs=CRS.from_epsg(code=32648),
                                                    dtype=str(pr_gradient.dtype))
    
                        new_dataset.write(pr_gradient.squeeze(), 1)
                        new_dataset.close()
            except IndexError:
                break

    def mosaic_predictions(self):
        raster_groups = dict()
        for _ in range(1000000):
            try:
                for i in range(len(self.test_generator.get_name(_))):
                    filename = self.test_generator.get_name(_)[i]
                    ref_number = filename.split("_")[3]
                    img = io.imread(os.path.join(self.prediction_dir, 'prediction_' + filename))
                    patch_dict = {"name": filename, "image": img}
                    if ref_number in raster_groups:
                        raster_groups[ref_number].append(patch_dict)
                    else:
                        raster_groups[ref_number] = [patch_dict]
            except IndexError:
                break

        for key in raster_groups:
            # for Google imagery processing this code is slow, multiprocessing could offer a solution
            # if key.startswith('26'):  # 16 is running, next = 21
            mosaic = None
            for image in self.images_list:
                if image.startswith(key):
                    raster_open = rasterio.open(os.path.join(self.test_dir, self.images_dir, image))
                    parent_img = reshape_as_image(raster_open.read())[:, :, 0]
                    extents = raster_open.bounds
                    transform = rasterio.transform.from_bounds(*extents, width=raster_open.width,
                                                               height=raster_open.height)
                    count = 0
                    for patch in raster_groups[key]:
                        base_img = np.empty(parent_img.shape)
                        base_img[:] = np.nan
                        c_off = patch.get("name").split("_")[1]
                        r_off = patch.get("name").split("_")[2]
                        d_arr = patch.get("image")
                        base_img[int(r_off): int(r_off) + self.image_size,
                        int(c_off): int(c_off) + self.image_size] = d_arr

                        if count == 0:
                            mosaic = base_img
                        else:
                            mosaic = np.dstack((mosaic, base_img))
                            mosaic = np.nanmin(mosaic, axis=2)
                        count += 1

            with rasterio.open(os.path.join(self.mosaic_dir, key + '.tif'),
                               mode='w',
                               driver='GTiff',
                               height=mosaic.shape[0],
                               width=mosaic.shape[1],
                               transform=transform,
                               count=1,
                               dtype='float32',
                               crs=CRS.from_epsg(32648)
                               ) as dest:
                dest.write(mosaic, 1)
                dest.close()

    def georeference_watershed(self):
        for file in self.images_list:
            raster_open = rasterio.open(os.path.join(self.test_dir, self.images_dir, file))
            extents = raster_open.bounds
            transform = rasterio.transform.from_bounds(*extents, width=raster_open.width, height=raster_open.height)
            with rasterio.open(os.path.join(self.watershed_dir,
                                            file.replace('_cambodia.tif',
                                                         '-watershed-lines.tif').replace('_vietnam.tif',
                                                                                         '-watershed-lines.tif')),
                               mode='r+') as src:
                image = src.read(1)
                image = image == 0
                image[image == 255] = src.nodata

                with rasterio.open(os.path.join(self.watershed_geo_dir,
                                                file.replace('_cambodia.tif',
                                                             '-watershed-lines.tif').replace('_vietnam.tif',
                                                                                             '-watershed-lines.tif')),
                                   mode='w',
                                   driver='GTiff',
                                   transform=transform,
                                   width=src.width,
                                   height=src.height,
                                   count=1,
                                   dtype='uint8',
                                   crs=CRS.from_epsg(32648)
                                   ) as dest:
                    dest.write(image, 1)
                    dest.close()
                    src.close()

    def evaluate_watershed(self):
        confusion_matrices = pd.DataFrame(columns=['ref_number', 'precision', 'recall', 'f1'])
        for file in self.masks_list:
            ref_number = file.split("_")[0]
            with rasterio.open(os.path.join(self.test_dir, self.masks_dir, file), 'r') as gt:
                arr_gt = gt.read()

            if self.source == Base.SOURCE.GOOGLE.value:
                with rasterio.open(os.path.join(self.watershed_expand_dir, f'{ref_number}.tif'), 'r') as ws:
                    arr_ws = ws.read()
                    arr_ws[arr_ws == 1] = 255
            else:
                with rasterio.open(os.path.join(self.watershed_dir, f'{ref_number}-watershed-lines.tif'), 'r') as ws:
                    arr_ws = ws.read()
                    arr_ws[arr_ws == 0] = 1
                    arr_ws[arr_ws == 255] = 0
                    arr_ws[arr_ws == 1] = 255

            labels = arr_gt.squeeze().flatten()
            predictions = arr_ws.squeeze().flatten()
            results = precision_recall_fscore_support(labels, predictions)
            bound_row = {'ref_number': ref_number,
                         'precision': results[0][1],
                         'recall': results[1][1],
                         'f1': results[2][1]}
            row_df = pd.DataFrame([bound_row])
            confusion_matrices = pd.concat([confusion_matrices, row_df], ignore_index=True)

        confusion_matrices.to_csv(
            f'output/evaluate/confusion_matrices_{self.source}_{self.area}_from_scratch_{self.from_scratch}.csv')

    def calculate_polis(self):
        vector_bounds = None

        for file in self.images_list:
            reference_gdf = None
            vector_gdf = None

            ref_dir = os.path.join(self.test_patches_dir, 'reference')
            vec_dir = os.path.join(self.test_patches_dir, 'vector')
            ref_number = str(file.split("_")[0])

            for x in os.listdir(ref_dir):
                if x.startswith(ref_number):
                    reference_gdf = gp.read_file(os.path.join(ref_dir, x), layer='reference')

            for y in os.listdir(vec_dir):
                if y.startswith(ref_number):
                    if y.endswith('shp'):
                        vector_gdf = gp.read_file(os.path.join(vec_dir, y))
                        bbox = vector_gdf.total_bounds
                        p1 = Point(bbox[0], bbox[3])
                        p2 = Point(bbox[2], bbox[3])
                        p3 = Point(bbox[2], bbox[1])
                        p4 = Point(bbox[0], bbox[1])
                        np1 = (p1.coords.xy[0][0], p1.coords.xy[1][0])
                        np2 = (p2.coords.xy[0][0], p2.coords.xy[1][0])
                        np3 = (p3.coords.xy[0][0], p3.coords.xy[1][0])
                        np4 = (p4.coords.xy[0][0], p4.coords.xy[1][0])
                        bb_polygon = Polygon([np1, np2, np3, np4])
                        vector_bounds = gp.GeoDataFrame(gp.GeoSeries(bb_polygon), columns=['geometry']) \
                            .set_crs(reference_gdf.crs)

            reference_gdf = reference_gdf.sjoin(vector_bounds, how='inner', predicate='within')

            polis_gdf = Polis().score(reference_gdf, vector_gdf)
            polis_gdf['polis'] = polis_gdf['polis'].astype(float)
            polis_gdf['area'] = polis_gdf['geometry'].area

            polis_gdf.to_file(os.path.join(self.polis_dir, f'{ref_number}_polis.gpkg'), driver="GPKG")

    def evaluate_polis(self):
        polis_list = os.listdir(self.polis_dir)
        polis_gdf_all = gp.GeoDataFrame()

        for polis_file in polis_list:
            ref_number = polis_file.split("_")[0]
            polis_gdf = gp.read_file(os.path.join(self.polis_dir, polis_file), layer=f'{ref_number}_polis')
            polis_gdf = polis_gdf.drop(columns='geometry')
            polis_gdf = polis_gdf.mean(axis=0)
            polis_row = {'ref_number': ref_number, 'polis': polis_gdf['polis']}

            polis_row_df = pd.DataFrame([polis_row])
            polis_gdf_all = pd.concat([polis_gdf_all, polis_row_df], ignore_index=True)

        polis_gdf_all.to_csv(f'output/evaluate/polis_{self.source}_{self.area}_from_scratch_{self.from_scratch}.csv')
