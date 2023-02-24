
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model.logistic import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras import initializers, Input, Model, regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, Lambda, Activation, Conv1D, MaxPooling1D, LSTM, \
    Concatenate, Conv2D, Reshape, MaxPooling2D, Flatten, AveragePooling2D, add, Permute, Layer
import tensorflow.keras.backend as K

from myconfig import Model_config

config = Model_config()

class AttentionLayer(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.get('normal')
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        print(len(input_shape))
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name="attention_weight")
        self.b = K.variable(self.init((self.attention_dim,)), name="attention_bias")
        self.u = K.variable(self.init((self.attention_dim, 1)), name="attention_vector")
        super(AttentionLayer, self).build(input_shape)

    def get_config(self):
        config = {
            'attention_dim': self.attention_dim
        }
        base_config = super(AttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        uit = K.relu(K.bias_add(K.dot(x, self.W), self.b))

        ait = K.exp(
            K.squeeze(
                K.dot(
                    uit, self.u) / self.attention_dim ** 0.5, -1))

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) +
                      K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(ait)

        output = K.sum(weighted_input, axis=1)

        return output, ait

def MMADL(feature_dim):
    branch_output = []
    branch_input = []

    if config.m_type == "dnn":
        for i in range(len(feature_dim)):
            input_i = Input(
                shape=(feature_dim[i] * 2,), name="input" + '_' + str(i + 1))
            # the first branch operates on the first input
            train_in_i = Dense(
                512,
                activation='relu',
                kernel_regularizer=regularizers.l2(1e-8))(input_i)
            train_in_i = BatchNormalization()(train_in_i)
            train_in_i = Dropout(config.dropout_rate)(train_in_i)
            train_in_i = Dense(
                256,
                activation='relu',
                kernel_regularizer=regularizers.l2(1e-8))(train_in_i)
            train_in_i = BatchNormalization()(train_in_i)
            train_in_i = Dropout(config.dropout_rate)(train_in_i)
            out_i = Dense(
                256,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-8))(train_in_i)
            branch_i = Model(inputs=input_i, outputs=out_i)
            branch_output.append(branch_i.output)
            branch_input.append(branch_i.input)
    else:
        for i in range(len(feature_dim)):
            input_i = Input(
                shape=(feature_dim[i] * 2,), name="input" + '_' + str(i + 1))
            train_in_i = Reshape(
                input_shape=(
                    feature_dim[i] * 2,
                ),
                target_shape=(
                    4,
                    feature_dim[i] // 2,
                    1))(input_i)
            print(input_i.shape)
            train_in_i = Conv2D(
                filters=16,
                kernel_size=3,
                activation="relu")(train_in_i)
            print(train_in_i.shape)
            train_in_i = AveragePooling2D()(train_in_i)
            print(train_in_i.shape)
            train_in_i = Flatten()(train_in_i)
            out_i = Dense(
                256,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-8))(train_in_i)
            branch_i = Model(inputs=input_i, outputs=out_i)
            branch_output.append(branch_i.output)
            branch_input.append(branch_i.input)

    stacked_tensor = Lambda(lambda x: K.stack(x, axis=1))(branch_output)

    after_attention = AttentionLayer(attention_dim=config.attention_dim)(stacked_tensor)

    train_in_i = Dense(config.event_num, activation='relu')(after_attention)
    train_in_i = BatchNormalization()(train_in_i)
    train_in_i = Dropout(config.dropout_rate)(train_in_i)
    train_in_i = Dense(config.event_num)(train_in_i)

    y = Activation('softmax')(train_in_i)

    model = Model(inputs=branch_input, outputs=y)
    Adam = optimizers.adam(learning_rate=config.lr)
    model.compile(
        optimizer=Adam,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.summary()
    return model


class Self_Attention(Layer):
    def __init__(self, attention_dim):
        super(Self_Attention, self).__init__()
        self.attention_dim = attention_dim
        self.wq = Dense(attention_dim)
        self.wk = Dense(attention_dim)
        self.wv = Dense(attention_dim)

    def scaled_dot_product_attention(self, q, k, v):

        k = Permute((2, 1))(k)
        print(q.shape, k.shape)
        matmul_qk = K.batch_dot(q, k, axes=(2, 1))
        print(matmul_qk.shape)

        dk = K.cast(k.shape[-1], dtype='float32')
        scaled_attention_logits = matmul_qk / K.sqrt(dk)

        attention_weights = K.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        print(attention_weights)
        output = K.batch_dot(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def compute_output_shape(self, input_shape):
        print((input_shape[0], input_shape[-2], self.attention_dim))
        return input_shape[0], input_shape[-2], self.attention_dim

    def call(self, x):
        batch_size = K.shape(x)[0]
        q = self.wq(x)  # (batch_size, seq_len, attention_dim)
        k = self.wk(x)  # (batch_size, seq_len, attention_dim)
        v = self.wv(x)  # (batch_size, seq_len, attention_dim)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
        output = scaled_attention  # (batch_size, seq_len_q, attention_dim)
        return output

def DeepDDI():

    model_input = Input(shape=(config.vector_size * 2,), name="input")
    for i in range(0, 8):
        output = Dense(2048, activation='relu')(model_input)
        output = BatchNormalization()(output)
    output = Dense(config.event_num)(output)
    output = Activation('softmax')(output)
    model = Model(inputs=model_input, outputs=output)
    Adam = optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=Adam,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.summary()
    return model


def novelDeepDDI():

    hidden_neurons = [500, 200, 500]
    inputA = Input(shape=(config.vector_size * 2,), name="inputA")
    inputB = Input(shape=(config.vector_size * 2,), name="inputB")
    # embedding input
    inputC = Input(shape=(config.embedding_size * 2,), name="inputC")

    # autoencoder_a
    encoder_a = Dense(hidden_neurons[0], activation='relu')(inputA)
    encoder_a = BatchNormalization()(encoder_a)
    encoder_a = Dropout(config.dropout_rate)(encoder_a)
    encoder_a = Dense(hidden_neurons[1], activation='relu')(encoder_a)
    encoder_a = BatchNormalization()(encoder_a)
    encoder_a = Dropout(config.dropout_rate)(encoder_a)
    decoder_a = Dense(hidden_neurons[2], activation='relu')(encoder_a)
    decoder_a = BatchNormalization()(decoder_a)
    decoder_a = Dropout(config.dropout_rate)(decoder_a)
    decoder_a = Dense(config.vector_size * 2, activation='relu')(decoder_a)

    # autoencoder_b
    encoder_b = Dense(hidden_neurons[0], activation='relu')(inputB)
    encoder_b = BatchNormalization()(encoder_b)
    encoder_b = Dropout(config.dropout_rate)(encoder_b)
    encoder_b = Dense(hidden_neurons[1], activation='relu')(encoder_b)
    encoder_b = BatchNormalization()(encoder_b)
    encoder_b = Dropout(config.dropout_rate)(encoder_b)
    dencoder_b = Dense(hidden_neurons[2], activation='relu')(encoder_b)
    dencoder_b = BatchNormalization()(dencoder_b)
    dencoder_b = Dropout(config.dropout_rate)(dencoder_b)
    dencoder_b = Dense(config.vector_size * 2, activation='relu')(dencoder_b)

    # autoencoder_c
    encoder_c = Dense(hidden_neurons[0], activation='relu')(inputB)
    encoder_c = BatchNormalization()(encoder_c)
    encoder_c = Dropout(config.dropout_rate)(encoder_c)
    encoder_c = Dense(hidden_neurons[1], activation='relu')(encoder_c)
    encoder_c = BatchNormalization()(encoder_c)
    encoder_c = Dropout(config.dropout_rate)(encoder_c)
    decoder_c = Dense(hidden_neurons[2], activation='relu')(encoder_c)
    decoder_c = BatchNormalization()(decoder_c)
    decoder_c = Dropout(config.dropout_rate)(decoder_c)
    decoder_c = Dense(config.embedding_size * 2, activation='relu')(decoder_c)



def Conv_LSTM():

    input_layer = Input(shape=(config.embedding_size * 2,))
    train_in_i = Reshape(input_shape=(config.embedding_size * 2,), target_shape=(1, config.embedding_size * 2))(
        input_layer)
    # convolution
    conv1 = Conv1D(
        filters=32,
        kernel_size=8,
        strides=2,
        activation='relu',
        padding='same')(train_in_i)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=2, strides=2, padding='same')(bn1)

    conv2 = Conv1D(
        filters=32,
        kernel_size=4,
        strides=2,
        activation='relu',
        padding='same')(pool1)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=2, strides=2, padding='same')(bn2)

    # lstm
    lstm1 = LSTM(128, return_sequences=True)(bn2)
    do3 = Dropout(0.5)(lstm1)

    lstm2 = LSTM(64)(do3)
    do4 = Dropout(0.2)(lstm2)

    output_layer = Dense(config.event_num, activation='softmax')(do4)

    model = Model(inputs=input_layer, outputs=output_layer)

    Adam = optimizers.Adam(lr=0.01)
    model.compile(
        optimizer=Adam,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.summary()

    return model


def MLP(parameter):
    # input layer
    model_input = Input(shape=(parameter["embedding_dim"] * 2,), name="input")

    # hidden layer
    for node_nums in parameter["layers"]:
        hidden_x = Dense(node_nums, activation='relu')(model_input)
        hidden_x = BatchNormalization()(hidden_x)
        hidden_x = Dropout(parameter["dropout"])(hidden_x)

    # output_layer
    hidden_x = Dense(parameter["event_num"], activation="relu")(hidden_x)
    y = Activation('softmax')(hidden_x)

    model = Model(inputs=model_input, outputs=y)
    Adam = optimizers.Adam(learning_rate=parameter["initial_lr"])
    model.compile(
        optimizer=Adam,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # model.summary()
    return model


def MMADL(feature_dim):
    branch_input = None
    if len(feature_dim) == 1:
        # single feature
        branch_input = Input(
            shape=(feature_dim[0] * 2,), name="input" + '_' + str(feature_dim[0]))
        x = Dense(
            config.hidden_dim,
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-8))(branch_input)
        x = BatchNormalization()(x)
        fusion_feature = Dropout(config.dropout_rate)(x)

    else:
        # multiple feature
        branch_output = []
        branch_input = []
        # 维度统一
        for i in range(len(feature_dim)):
            input_i = Input(
                shape=(feature_dim[i] * 2,), name="input" + '_' + str(i + 1))

            # 单层MLP 进行特征映射
            # [128, 256, 512]
            high_input_i = Dense(
                config.hidden_dim,
                activation='relu',
                kernel_regularizer=regularizers.l2(1e-8))(input_i)
            high_input_i = BatchNormalization()(high_input_i)
            out_i = Dropout(config.dropout_rate)(high_input_i)

            # 模型搭建
            branch_i = Model(inputs=input_i, outputs=out_i)
            branch_output.append(branch_i.output)
            branch_input.append(branch_i.input)

        if config.fusion_type == "concate":
            # Concatenate
            fusion_feature = Concatenate()([branch for branch in branch_output])
            # print(Concatenate_tensor.shape)
        elif config.fusion_type == "sum":
            # mean
            Sum_tensor = add([branch for branch in branch_output])
            fusion_feature = Lambda(lambda x: (x / len(feature_dim)))(Sum_tensor)
            print(fusion_feature.shape)
        elif config.fusion_type == "self_attention":
            # reshape
            stacked_tensor = Lambda(lambda x: K.stack(x, axis=1))(branch_output)
            print(stacked_tensor.shape)
            # self attention:
            fusion_feature = Self_Attention(attention_dim=config.attention_dim)(stacked_tensor)
            # fusion_feature = Lambda(lambda x: K.flatten(x))(fusion_feature)
            fusion_feature = Reshape((fusion_feature.shape[-1]*fusion_feature.shape[-2],))(fusion_feature)
            print(fusion_feature.shape)
        else:
            if config.fusion_type == "attention_and_concate":
                stacked_tensor = Lambda(lambda x: K.stack(x, axis=1))(branch_output[0:-1])
                print(stacked_tensor.shape)
                fusion_feature = AttentionLayer(attention_dim=config.attention_dim)(stacked_tensor)
                # cocate local and global feature
                fusion_feature = Concatenate()([fusion_feature, branch_output[-1]])

            elif config.fusion_type == "attention":

                stacked_tensor = Lambda(lambda x: K.stack(x, axis=1))(branch_output)
                print(stacked_tensor.shape)
                fusion_feature, score = AttentionLayer(attention_dim=config.attention_dim)(stacked_tensor)

    # MLP
    hidden_x = None
    for i in range(len(config.MLP_layers)):
        u_nums = config.MLP_layers[i]
        if i == 0:
            hidden_x = Dense(
                u_nums,
                activation='relu',
                kernel_regularizer=regularizers.l2(1e-8), name="MLP_" + str(i + 1))(fusion_feature)
        else:
            hidden_x = Dense(
                u_nums,
                activation='relu',
                kernel_regularizer=regularizers.l2(1e-8), name="MLP_" + str(i + 1))(hidden_x)
        hidden_x = BatchNormalization(name="MLP_BN_" + str(i + 1))(hidden_x)
        hidden_x = Dropout(config.dropout_rate, name="MLP_DP_" + str(i + 1))(hidden_x)

    hidden_x = Dense(config.event_num)(hidden_x)
    y = Activation('softmax')(hidden_x)

    model = Model(inputs=branch_input, outputs=y)
    Adam = optimizers.Adam(learning_rate=config.lr)
    model.compile(
        optimizer=Adam,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.summary()

    return model


# Random Forest
# K-nearest Neighbors
if __name__ == '__main__':
    pass
