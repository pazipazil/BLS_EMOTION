开发环境：window11+python3.11+RTX4060

========== 核心必装库 ==========
mediapipe>=0.9.0
numpy>=1.21.0
opencv-python>=4.5.5
pandas>=1.4.0
scikit-learn>=1.0.2
pillow>=9.0.0

========== 可选扩展库（根据需求注释/取消注释） ==========
torch>=1.10.0          # 深度学习模型推理/训练
torchvision>=0.11.0    # PyTorch视觉配套工具
matplotlib>=3.5.0      # 可视化
seaborn>=0.11.0        # 增强可视化
joblib>=1.1.0          # 模型保存/加载
scipy>=1.7.0           # 高级数值计算




数据准备：RAF-DB 对齐图片、train_blendshapes.npz/test_blendshapes.npz 生成方式。

训练命令示例：
python extract_rafdb1_basic_blendshapes.py
python bls_train.py（52 维）
python train_mnetv3_small_solo.py → 导出特征
python bls_train_mnetv3.py（图像特征+BLS）

推理命令：
python face.detect.py
python face_detect_mnetv3.py

先运行bls_train.py或者bls_train_mnetv3.py生成对应BLS模型，再运行face.detect.py或者face_detect_mnetv3.py，目前face_detect.py的精度比较高
