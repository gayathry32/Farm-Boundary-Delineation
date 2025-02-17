# import osgeo
import os
import pandas as pd
from deep_learning_models.models.unet import Unet
from base import Base


class Evaluate(Base):

    def __init__(self, validate_generator):
        super().__init__()
        self.evaluate_model = Unet().load_model()
        self.validate_generator = validate_generator
        self.validation_steps = len(
            os.listdir(os.path.join(self.validate_patches_dir, self.images_dir))) // self.batch_size

    def evaluate_models(self):       
        scores = {}       
        for f in os.listdir(self.weights_dir):
            if self.area in f and self.source in f and str(self.from_scratch) in f:
                weights_path = os.path.join(self.weights_dir, f)
                self.evaluate_model.load_weights(weights_path)
                scores[f] = self.evaluate_model.evaluate(self.validate_generator,
                                                         verbose=1,
                                                         use_multiprocessing=True,
                                                         batch_size=self.batch_size,
                                                         steps=self.validation_steps)

        df = pd.DataFrame.from_dict(scores, orient="index")
        df.to_csv(f'output/validate/test_{self.source}_{self.area}_from_scratch_{self.from_scratch}.csv')
