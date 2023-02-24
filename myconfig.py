class Model_config(object):
    def __init__(self):
        self.epochs = 200
        self.is_row_feature = False
        self.feature_set_length = None
        self.event_num = 65  # 分类数目
        self.dropout_rate = 0.3  # 丢弃率
        self.vector_size = 572  # 向量的size

        # self.attention_dim = 512
        self.attention_dim = 128  # 注意力向量的维度
        # self.attention_dim = 64
        # self.attention_dim = 32

        self.embedding_size = 400  # kg_embedding的维度

        self.hidden_dim = 64
        # self.hidden_dim = 1024
        # self.hidden_dim = 2048

        # self.fusion_type = "attention_and_concate"
        # self.fusion_type = "sum"
        # self.fusion_type = "concate"
        # self.fusion_type = "self_attention"
        self.fusion_type = "attention"


        # self.MLP_layers = [512, 256, 256, 128, 128]
        self.MLP_layers = [2048, 1024, 512]
        # self.MLP_layers = [1024,512, 256]
        # self.MLP_layers = [512, 256, 128]
        self.monitor = "val_accuracy"
        self.embedding_model = "complex"
        self.embedding_path = "embedding_data\ComplEx_embedding_dict_200_0.6025.csv"
        self.batch_size = 128
        self.up_sampling_rate = 5
        self.lr = 0.0005
        # self.lr = 0.001
        self.dataset = "sub_drkg"
        self.label_data_filename = "data_label.npy"
        self.attention_head = 2
        self.m_type = "dnn"
