import numpy as np

# ------------------------------
# 1. BLS核心类（实现训练和预测逻辑）
# ------------------------------
class BLS:
    def __init__(self, n_input=8, n_feature=10, n_enhance=5, lambda_=0.01):
        """
        初始化BLS参数
        :param n_input: 输入特征维度（这里是8维）
        :param n_feature: 特征节点数（10）
        :param n_enhance: 增强节点数（5）
        :param lambda_: 正则化参数（0.01）
        """
        self.n_input = n_input
        self.n_feature = n_feature
        self.n_enhance = n_enhance
        self.lambda_ = lambda_
        
        # 待训练/随机生成的参数
        self.Wz = None  # 特征节点权重（n_feature × n_input）
        self.bz = None  # 特征节点偏置（n_feature × 1）
        self.Wh = None  # 增强节点权重（n_enhance × n_feature）
        self.bh = None  # 增强节点偏置（n_enhance × 1）
        self.W = None   # 输出权重（n_feature + n_enhance × 4）

    def relu(self, x):
        """ReLU激活函数：x>0保留，否则为0"""
        return np.maximum(x, 0)

    def train(self, X_train, Y_train):
        """
        训练BLS模型
        :param X_train: 训练特征（20×8矩阵）
        :param Y_train: 训练标签（20×4 one-hot矩阵）
        """
        # Step 1: 随机生成特征节点权重Wz和偏置bz（0~0.5均匀分布）
        np.random.seed(42)  # 固定随机种子，确保结果可复现
        self.Wz = np.random.uniform(0, 0.5, (self.n_feature, self.n_input))
        self.bz = np.random.uniform(0, 0.2, (self.n_feature,))

        # Step 2: 计算特征节点Z（20×10）
        Z = self.relu(np.dot(X_train, self.Wz.T) + self.bz)  # X(20×8) × Wz.T(8×10) = 20×10，加bz广播

        # Step 3: 随机生成增强节点权重Wh和偏置bh（0~0.5均匀分布）
        self.Wh = np.random.uniform(0, 0.5, (self.n_enhance, self.n_feature))
        self.bh = np.random.uniform(0, 0.2, (self.n_enhance,))

        # Step 4: 计算增强节点H（20×5）
        H = self.relu(np.dot(Z, self.Wh.T) + self.bh)  # Z(20×10) × Wh.T(10×5) = 20×5

        # Step 5: 拼接联合特征矩阵A（20×15）
        A = np.hstack([Z, H])

        # Step 6: 计算输出权重W（15×4），BLS唯一训练的参数
        A_T = A.T
        # 公式：W = (A^T*A + λI)^(-1) * A^T * Y
        I = np.eye(A.shape[1])  # 15×15单位矩阵
        ATA_lambdaI = np.dot(A_T, A) + self.lambda_ * I
        ATA_lambdaI_inv = np.linalg.pinv(ATA_lambdaI)  # 伪逆计算（避免矩阵奇异）
        self.W = np.dot(np.dot(ATA_lambdaI_inv, A_T), Y_train)

        print("BLS训练完成！")
        print(f"输出权重W维度：{self.W.shape}")

    def predict(self, X_new):
        """
        预测新样本情绪
        :param X_new: 新样本特征（1×8矩阵）
        :return: 情绪得分向量、预测情绪索引、情绪名称
        """
        if self.W is None:
            raise ValueError("请先训练模型！")

        # Step 1: 计算新样本的特征节点Z_new（1×10）
        Z_new = self.relu(np.dot(X_new, self.Wz.T) + self.bz)

        # Step 2: 计算新样本的增强节点H_new（1×5）
        H_new = self.relu(np.dot(Z_new, self.Wh.T) + self.bh)

        # Step 3: 拼接联合特征A_new（1×15）
        A_new = np.hstack([Z_new, H_new])

        # Step 4: 计算情绪得分（1×4）
        scores = np.dot(A_new, self.W)

        # Step 5: 确定预测情绪（得分最高的类别）
        emotion_idx = np.argmax(scores)
        emotion_names = ["高兴", "难过", "生气", "中性"]
        predict_emotion = emotion_names[emotion_idx]

        return scores, emotion_idx, predict_emotion

# ------------------------------
# 2. 准备训练数据（20个样本，8维特征+4类情绪）
# ------------------------------
def prepare_training_data():
    # 训练特征X：20×8矩阵（每类情绪5个样本，按之前设计的规律）
    X_train = np.array([
        # 5个高兴（前3维高，后5维低）
        [0.85,0.82,0.88,0.15,0.12,0.18,0.11,0.13],
        [0.81,0.87,0.83,0.11,0.16,0.14,0.13,0.15],
        [0.89,0.84,0.86,0.13,0.14,0.12,0.15,0.11],
        [0.83,0.85,0.81,0.16,0.11,0.15,0.12,0.14],
        [0.87,0.83,0.89,0.12,0.15,0.13,0.14,0.12],
        # 5个难过（中间3维高，前后低）
        [0.14,0.12,0.16,0.83,0.87,0.82,0.13,0.11],
        [0.11,0.15,0.13,0.86,0.81,0.85,0.15,0.12],
        [0.15,0.13,0.14,0.82,0.85,0.87,0.11,0.13],
        [0.12,0.16,0.15,0.87,0.83,0.81,0.12,0.15],
        [0.13,0.14,0.11,0.81,0.86,0.83,0.14,0.11],
        # 5个生气（后3维高，前5维低）
        [0.11,0.13,0.12,0.15,0.14,0.84,0.88,0.82],
        [0.13,0.11,0.15,0.12,0.16,0.87,0.83,0.85],
        [0.12,0.15,0.13,0.14,0.11,0.82,0.86,0.87],
        [0.15,0.12,0.14,0.16,0.13,0.85,0.81,0.83],
        [0.14,0.13,0.11,0.11,0.15,0.83,0.87,0.81],
        # 5个中性（所有维度0.5左右）
        [0.51,0.53,0.52,0.54,0.51,0.53,0.52,0.54],
        [0.53,0.52,0.54,0.51,0.53,0.52,0.54,0.51],
        [0.52,0.54,0.51,0.53,0.52,0.54,0.51,0.53],
        [0.54,0.51,0.53,0.52,0.54,0.51,0.53,0.52],
        [0.51,0.52,0.53,0.54,0.52,0.51,0.52,0.53]
    ])

    # 训练标签Y：20×4 one-hot矩阵（高兴=0, 难过=1, 生气=2, 中性=3）
    Y_train = np.array([
        [1,0,0,0]]*5 +  # 高兴
        [[0,1,0,0]]*5 +  # 难过
        [[0,0,1,0]]*5 +  # 生气
        [[0,0,0,1]]*5    # 中性
    )

    print("训练数据准备完成！")
    print(f"训练特征X维度：{X_train.shape}")
    print(f"训练标签Y维度：{Y_train.shape}")
    return X_train, Y_train

# ------------------------------
# 3. 交互式验证：输入新8维特征，预测情绪
# ------------------------------
def interactive_prediction(bls_model):
    while True:
        try:
            # 读取用户输入
            input_str = input("\n请输入8维特征（用空格分隔，输入q退出）：")
            if input_str.lower() == "q":
                print("退出预测！")
                break

            # 转换为8维数组
            X_new = np.array([float(x.strip()) for x in input_str.split()])
            if len(X_new) != 8:
                print("错误：请输入8个数值！")
                continue

            # 预测
            scores, emotion_idx, predict_emotion = bls_model.predict(X_new.reshape(1, -1))  # 转为1×8矩阵

            # 输出结果
            print("\n预测结果：")
            print(f"情绪得分向量（高兴/难过/生气/中性）：{np.round(scores, 3)}")
            print(f"预测情绪索引：{emotion_idx}")
            print(f"最终预测情绪：{predict_emotion}")

        except ValueError:
            print("错误：请输入有效的数字！")
        except Exception as e:
            print(f"错误：{str(e)}")

# ------------------------------
# 4. 主程序（训练+预测）
# ------------------------------
if __name__ == "__main__":
    # 步骤1：准备训练数据
    X_train, Y_train = prepare_training_data()

    # 步骤2：初始化并训练BLS模型
    bls = BLS(
        n_input=8,    # 8维输入特征
        n_feature=10, # 10个特征节点
        n_enhance=5,  # 5个增强节点
        lambda_=0.01  # 正则化参数
    )
    bls.train(X_train, Y_train)

    # 步骤3：交互式验证新特征
    interactive_prediction(bls)
