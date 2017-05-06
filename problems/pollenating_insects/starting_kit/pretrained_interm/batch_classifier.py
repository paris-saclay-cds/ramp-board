from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam, SGD

class BatchClassifier(object):
    
    def __init__(self):
        self.model = build_model()
    
    def fit(self, gen_builder):
        gen_train, gen_valid, nb_train, nb_valid = gen_builder.get_train_valid_generators(batch_size=16, valid_ratio=0.1)
        self.model.fit_generator(
                gen_train,
                samples_per_epoch=nb_train,
                nb_epoch=3,
                # In parallel to training, a CPU process loads and preprocesses data from disk and put
                # it into a queue in the form of mini-batches of size `batch_size`.`max_q_size` controls 
                # the maximum size of that queue.
                # The size of the queue should be big enough so that the training process (GPU) never
                # waits for data (the queue should be never be empty). 
                # The CPU process loads chunks of 1024 images each time, and
                # 1024/batch_size mini-batches from that chunk are put into the queue.
                # Assuming training the model on those 1024/batch_size mini-batches is slower than 
                # loading a single chunk of 1024 images, a good lower bound for `max_q_size` would be
                # (1024/batch_size). if `batch_size` is 16, you can put `max_q_size` to 64.
                max_q_size=16,
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
                # processes so that they load different data, this can become a bit 
                # complicated, so I choose to rather load exactly one chunk at a time using 
                # 1 worker (so `nb_worker` have to be equal to 1), but do this single
                # chunk loading in parallel with joblib.
                nb_worker=1,
                # if pickle_safe is True, processes are used instead of threads.
                # here, 1 process is used because `nb_worker` is 1.
                pickle_safe=True,
                validation_data=gen_valid,
                nb_val_samples=nb_valid,
                verbose=1)

    def predict_proba(self, X):
        return self.model.predict(X)
        
def build_model():
    vgg16 = VGG16(include_top=False, weights='imagenet')
    vgg16.trainable = False
    inp = vgg16.get_layer(name='input_1')
    hid = vgg16.get_layer(name='block3_conv3')
    vgg16_hid = Model(inp.input, hid.output)

    inp = Input((3, 224, 224))
    x = vgg16_hid(inp)
    x = Flatten(name='flatten')(x)
    x = Dense(500, activation='relu', name='fc')(x)
    out = Dense(18, activation='softmax', name='predictions')(x)
    model = Model(inp, out)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-4, momentum=0.95), metrics=['accuracy'])
    return model
