<!-- 数据准备：RAF-DB 对齐图片、train_blendshapes.npz/test_blendshapes.npz 生成方式。
训练命令示例：
python extract_rafdb1_basic_blendshapes.py
python bls_train.py（52 维）
python train_mnetv3_small_solo.py → 导出特征
python bls_train_mnetv3.py（图像特征+BLS）
推理命令：
python face.detect.py
python face_detect_mnetv3.py -->
