# 加密查重模块概况
基于TenSEAL中CKKS全同态加密实现方案搭建加密查重模块，共包含3个模块，即三个python脚本文件
# 操作指南
data文件夹下暂时存储验证的场景库编码后的数据，以及一个供验证的明文编码场景文件；keys文件夹下存储公钥与私钥；query文件夹下存储加密后的编码场景文件；results文件夹下存储密文计算的结果；

- step1：数据拥有方（用户端），运行module1_user.py脚本文件，生成公钥私钥文件以及密文文件，公钥+密文传递给平台（context_*_full.ctx，query_*_encrypted.npz），私钥存储在用户本地

- Step2：平台方，运行module2_platform.py脚本文件，读取密文，计算与场景库之间的距离，保存密文计算结果（仍然是密文形式），生成密文计算结果文件（*_candidates.txt，*.result.npz）

- Step3：数据拥有方（用户端），密文结果使用用户本地私钥解密并进行排序生成结果csv文件（*_final_similarity_result.csv）
备注：密文与单次生成的私钥和公钥是匹配的
