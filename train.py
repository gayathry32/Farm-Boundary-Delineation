import osgeo
import tensorflow as tf
from base import *
from datetime import datetime
from deep_learning_models.models.unet import Unet


class Train(Base):
    def __init__(self, train_generator, validate_generator):
        super().__init__()
        self.train_generator = train_generator
        self.validate_generator = validate_generator
        self.steps_per_epoch = len(os.listdir(os.path.join(self.train_patches_dir, self.images_dir))) // self.batch_size
        self.validation_steps = len(
            os.listdir(os.path.join(self.validate_patches_dir, self.images_dir))) // self.batch_size
        self.model_name = f'{self.model}_'\
                        f'{self.area}_'\
                        f'{self.source}_'\
                        f'from_scratch_{str(self.from_scratch)}_'\
                        f'{self.batch_size}_{self.steps_per_epoch}_'\
                        f'{self.validation_steps}_'\
                        f'{self.patience}_'\
                        f'{str(self.learning_rate).replace("0.", "")}_'\
                        f'{datetime.now().strftime("%Y%m%d-%H%M%S")}.h5'

            
    def _callbacks(self):
        return [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.weights_dir,
                             self.model_name),
                monitor='val_loss', save_best_only=True, mode='min'),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, min_delta=0, patience=self.patience)
        ]

    def _run_model(self, model, train_generator, validate_generator):
        return model.fit(
            train_generator,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            batch_size=self.batch_size,
            verbose=1,
            validation_data=validate_generator,
            validation_steps=self.validation_steps,
            shuffle=True,
            callbacks=self._callbacks(),
        )

    def train(self):
        unet = Unet()
        model = unet.load_model()

        if self.from_scratch:
            self._run_model(model, self.train_generator, self.validate_generator)

        elif not self.from_scratch:
            model.load_weights(os.path.join(self.weights_dir, self.pre_trained_weights_file_s2))
            self._run_model(model, self.train_generator, self.validate_generator)

