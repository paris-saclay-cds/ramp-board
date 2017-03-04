from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import Adam

class Classifier(object):
    
    def __init__(self):
        self.model = build_model()
    
    def fit(self, gen_builder):
        gen_train, gen_valid, nb_train, nb_valid = gen_builder.get_train_valid_generators(batch_size=64, valid_ratio=0.1)
        self.model.fit_generator(
                gen_train,
                samples_per_epoch=nb_train,
                nb_epoch=30,
                # `max_q_size` Should be large enough otherwise GPU will not be
                # used (1024 should be fine).
                max_q_size=1024,
                # WARNING : It is obligatory to set `nb_worker` to 1.
                # This in principle controls the number of workers used
                # by keras to load mini-batches from disk to memory in parallel
                # to GPU training. But I don't like the way it works and their
                # code is not very commented/used, so I dont trust it that much
                # (we might have surprises).
                # The way it works in keras is by launching in parallel `nb_worker` 
                # threads or processes which will all use a copy of the generator passed.
                # to `fit_generator`. So if nothing is done and `nb_worker` is set to 
                # some number > 1, the neural net will be trained with repetitions 
                # of the same data, because the workers are independent and they got 
                # through the same generator. 
                # Hence it is necessary to introduce a shared lock between the the 
                # processes so that they  load different data, this can become a bit 
                # complicated, so I choose to rather load exactly one chunk at a time using 
                # 1 worker (so `nb_worker` have to be equal to 1), but do this single
                # chunk loading in parallel with joblib.
                nb_worker=1,
                pickle_safe=True,
                validation_data=gen_valid,
                nb_val_samples=nb_valid,
                verbose=1)

    def predict_proba(self, X):
        return self.model.predict(X)
        
def build_model():
    # This is VGG16
    inp = Input((3, 64, 64))
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(inp)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # dense
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    out = Dense(18, activation='softmax', name='predictions')(x)
    model = Model(inp, out)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    return model
