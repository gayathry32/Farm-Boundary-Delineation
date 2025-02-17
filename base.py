import os
from pathlib import Path
from strenum import StrEnum

class Area(StrEnum):
    ASIA = 'asia'
    NL = 'nl'

class Source(StrEnum):
    S2 = 's2'

class SelectedModel(StrEnum):
    SATELLITE_UNET = 'satellite_unet'
    CUSTOM_UNET = 'custom_unet'

class Base:
    # config------------------------------------------------------------------------------------------------------------
    INPUT_DIR = r'/content/drive/MyDrive/AI4SmallFarms'
    AREA = Area.ASIA
    SOURCE = Source.S2
    IMAGE_SIZE = 256
    NUM_CHANNELS = 3

    MODEL = SelectedModel.SATELLITE_UNET
    FROM_SCRATCH = True
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    EPOCHS = 1000 #at least 500
    PATIENCE = 10
    BIN_THRESHOLD = .5 

    BEST_NL_S2 = r'best/linux_s2_nl_from_scratch_True_8_120_26_10_001_20230509-101508.h5'
    BEST_ASIA_S2 = r''
    BEST_ASIA_S2_SCRATCH = r'satellite_unet_asia_s2_from_scratch_True_8_31_7_10_0001_20240223-093924.h5'

    VISUALIZE_PREDICTIONS = True
    WRITE_PREDICTIONS = True

    # constants---------------------------------------------------------------------------------------------------------
    EXPERIMENT_NAME = 'ai4small'
    SPLIT_1 = .7
    SPLIT_2 = .85
    
    DATA_DIR = os.path.join(INPUT_DIR, AREA)
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VALIDATE_DIR = os.path.join(DATA_DIR, 'validate')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    IMAGES_DIR = 'images'
    MASKS_DIR = 'masks'
    
    PATCHES_DIR = 'patches'
    TRAIN_PATCHES_DIR = os.path.join(TRAIN_DIR, PATCHES_DIR)
    VALIDATE_PATCHES_DIR = os.path.join(VALIDATE_DIR, PATCHES_DIR)
    TEST_PATCHES_DIR = os.path.join(TEST_DIR, PATCHES_DIR)

    WEIGHTS_DIR = r'deep_learning_models/weights'
    PREDICT_DIR = 'predict'
    MODEL_DIR = 'deep_learning_models/saved_models'
    OUTPUT_DIRS = ['vector', 'watershed', 'gradient', 'prediction']

    def __init__(self):
        # config--------------------------------------------------------------------------------------------------------
        self.experiment_name = Base.EXPERIMENT_NAME
        self.area = Base.AREA
        self.source = Base.SOURCE
        self.image_size = Base.IMAGE_SIZE
        self.num_channels = Base.NUM_CHANNELS

        self.split_1 = Base.SPLIT_1
        self.split_2 = Base.SPLIT_2

        self.model = Base.MODEL
        self.from_scratch = Base.FROM_SCRATCH
        self.batch_size = Base.BATCH_SIZE
        self.learning_rate = Base.LEARNING_RATE
        self.epochs = Base.EPOCHS
        self.patience = Base.PATIENCE
        self.bin_threshold = Base.BIN_THRESHOLD

        self.best_nl_s2 = Base.BEST_NL_S2
        self.best_asia_s2 = Base.BEST_ASIA_S2
        self.best_asia_s2_scratch = Base.BEST_ASIA_S2_SCRATCH
        self.pre_trained_weights_file_s2 = Base.BEST_NL_S2

        self.visualize_predictions = Base.VISUALIZE_PREDICTIONS
        self.write_predictions = Base.WRITE_PREDICTIONS

        # constants-----------------------------------------------------------------------------------------------------
        self.data_dir = Base.DATA_DIR
        self.train_dir = Base.TRAIN_DIR
        self.validate_dir = Base.VALIDATE_DIR
        self.test_dir = Base.TEST_DIR
        self.images_dir = Base.IMAGES_DIR
        self.masks_dir = Base.MASKS_DIR
        self.input_dir = Base.INPUT_DIR

        self.patches_dir = Base.PATCHES_DIR
        self.train_patches_dir = Base.TRAIN_PATCHES_DIR
        self.validate_patches_dir = Base.VALIDATE_PATCHES_DIR
        self.test_patches_dir = Base.TEST_PATCHES_DIR

        self.weights_dir = Base.WEIGHTS_DIR
        self.predict_dir = Base.PREDICT_DIR
        self.model_dir = Base.MODEL_DIR
        self.output_dirs = Base.OUTPUT_DIRS

        # logic---------------------------------------------------------------------------------------------------------
        if not self.from_scratch and self.source == Source.S2 and self.area == Area.ASIA:
            self.predict_weights_file = Base.BEST_ASIA_S2
        elif self.from_scratch and self.source == Source.S2 and self.area == Area.ASIA:
            self.predict_weights_file = Base.BEST_ASIA_S2_SCRATCH
        elif self.from_scratch and self.source == Source.S2 and self.area == Area.NL:
            self.predict_weights_file = Base.BEST_NL_S2

        dirs = [
            os.path.join(self.train_dir, self.images_dir),
            os.path.join(self.train_dir, self.masks_dir),
            os.path.join(self.validate_dir, self.images_dir),
            os.path.join(self.validate_dir, self.masks_dir),
            os.path.join(self.test_dir, self.images_dir),
            os.path.join(self.test_dir, self.masks_dir)
        ]

        for folder in dirs:
            Path(folder).mkdir(parents=True, exist_ok=True)

        input_patch_dirs = [
            os.path.join(self.train_dir, self.patches_dir, self.images_dir),
            os.path.join(self.train_dir, self.patches_dir, self.masks_dir),
            os.path.join(self.validate_dir, self.patches_dir, self.images_dir),
            os.path.join(self.validate_dir, self.patches_dir, self.masks_dir),
            os.path.join(self.test_dir, self.patches_dir, self.images_dir),
            os.path.join(self.test_dir, self.patches_dir, self.masks_dir)
        ]

        for patch_folder in input_patch_dirs:
            Path(patch_folder).mkdir(parents=True, exist_ok=True)

        self.RESULT_DIR = os.path.join(self.input_dir, 'output')
        results_dir = [os.path.join(self.RESULT_DIR,'evaluate'), os.path.join(self.RESULT_DIR,'validate')]
        for folder in results_dir:
            Path(folder).mkdir(parents=True, exist_ok=True)
                
    def update_modelName(self, model_name):
        self.predict_weights_file = model_name