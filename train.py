from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop, Adam, Adagrad

import config
import utils
import model

total_train_images = utils.total_count_files(config.TRAIN_DIR)
print(total_train_images)

total_val_images = utils.total_count_files(config.TESTE_DIR)
print(total_val_images)


dataset_train = ImageDataGenerator(
                                rescale=1/255,
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest'
                                )

train_generator = dataset_train.flow_from_directory(config.TRAIN_DIR,
                                                    classes = ["0", "1"],
                                                    target_size=config.image_size_gen,
                                                    batch_size=config.batch,
                                                    class_mode='categorical')

valida_generator = dataset_train.flow_from_directory(config.TESTE_DIR,
                                                    classes = ["0", "1"],
                                                    target_size=config.image_size_gen,
                                                    batch_size=config.batch,
                                                    class_mode='categorical')


model = model.custom_inceptionResnetV2_conv_global()

#optimizer
optimizer=SGD(learning_rate=1e-3,
              momentum=0.9,
              nesterov=True)
optimizer_rms=RMSprop(learning_rate=0.0001, rho=0.9, epsilon=None, decay=0.0)

optimizer_adam=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

optimizer_adagrad=Adagrad(learning_rate=0.001, epsilon=None, decay=0.0)
    
objective="categorical_crossentropy"

model.compile(optimizer=optimizer_adam,
              loss=objective,
              metrics=['accuracy']
              )

# callbacks
history = utils.LossHistory()
early_stopping = utils.set_early_stopping()
#model_cp = model_utils.set_model_checkpoint()
reduce_lr = utils.set_reduce_lr()



steps_train = int(total_train_images // config.batch)
steps_val = int(total_val_images // config.batch)


# training model
history = model.fit(train_generator,
                            steps_per_epoch = steps_train,
                            epochs = config.epochs,
                            callbacks=[history, early_stopping, reduce_lr],
                            validation_data = valida_generator,
                            validation_steps = steps_val,
                            verbose = 2)

utils.save_model(model)


image = utils.load('./datasets/PROCESSED/TESTE/0/parte_1_171.png')
model.predict(image)
