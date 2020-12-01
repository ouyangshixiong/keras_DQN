from keras.models import Model
from keras.layers import Input, Dense
class Network:
    def __init__(self):
        """基本网络结构.
        """
        inputs = Input(shape=(4,))
        x = Dense(16, activation='relu')(inputs)
        x = Dense(16, activation='relu')(x)
        x = Dense(2, activation='linear')(x)
        self.model = Model(inputs=inputs, outputs=x)

    def build_model(self):
        return self.model
