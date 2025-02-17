import rasterio
import base
from datetime import datetime

from deep_learning_models.models.unet import Unet
from deep_learning_models.utils.data_generator import DataGenerator
from preprocess.split_dataset import SplitDataset
from preprocess.create_patches import CreatePatches
from train import Train
from evaluate import Evaluate
from predict import Predict


# https://www.youtube.com/watch?v=YJ1wSxbqqo8
# TODO: make config available here
# Three experiments:
# - s2 asia from pretrained X
# - s2 asia from scratch X
# - google asia from scratch expand raster reference lines

if __name__ == "__main__":
    config = base.Base()

    # # step 1: split dataset-------------------------------------------------------------------------------------------

    split_dataset_obj = SplitDataset()
    split_dataset_obj.split_dataset()

    # # step 2: create patches------------------------------------------------------------------------------------------

    create_patches_obj = CreatePatches()
    create_patches_obj.create_patch()

    train_generator = DataGenerator(config.train_patches_dir)
    validate_generator = DataGenerator(config.validate_patches_dir)
    test_generator = DataGenerator(config.test_patches_dir)
    
    # # step 3: create model--------------------------------------------------------------------------------------------

    create_obj = Unet().create_model()

    # # step 4: train model---------------------------------------------------------------------------------------------

    train_obj = Train(train_generator, validate_generator)
    train_obj.train()

    # # step 5: evaluate models-----------------------------------------------------------------------------------------

    evaluate_obj = Evaluate(validate_generator)
    evaluate_obj.evaluate_models()

    # # step 6 make predictions-----------------------------------------------------------------------------------------
    test_generator.update_modelName(train_obj.model_name)
    predict_obj = Predict(test_generator)
    predict_obj.create_predictions()
    predict_obj.mosaic_predictions()

    # # step 7 watershed segmentation and polygonaztion------------------------------------------------------------------
    # predict_obj.georeference_watershed()
    # predict_obj.evaluate_watershed()
    # predict_obj.calculate_polis()
    # predict_obj.evaluate_polis()
