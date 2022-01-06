**English** | [简体中文](./README.zh-CN.md)
<h1 align="center">Cnn-Classification-Dog-Vs-Cat 猫狗辨别</h1>

 (pytorch版本) CNN Resnet18 的猫狗分类器，基于ResNet及其变体网路系列，对于一般的图像识别任务表现优异，模型精准度高达93%（小型样本）。
![Resnet](https://github.com/yexiaopingguo/Cat-Dog-Classification/blob/main/resnet.png)

项目制作于本科大三学习 [認識人工智慧AI：企業人工智慧] 课堂期间，正好遇上本人对这方面感兴趣的阶段，所以选择了入门的深度学习项目练手，希望做为兴趣点激励自己学习！距离DDL只剩几天了，现阶段时间和精力有限，遂没有自建神经网络，只是利用了已训练的常用网络进行深度学习，以后找机会补上。

## 1 requirement
- python3
- matplotlib
- numpy
- pytorch
- pandas
- os
- Images

## 2 Description of files
- inputs: 包含猫狗训练和测试样本图片数据[[下载地址]](https://www.kaggle.com/c/dogs-vs-cats/data)，经过特殊改良，其中训练集包含1000笔狗狗图片、1000笔猫咪图片，测试集包含100笔猫狗混合图片；
- dog_cat_classcial.ipynb：主文件，训练后测试集精度约 93%
- ckpt_resnet18_catdog.pth：基于CNN的预测模型
- preds_resnet18.csv：预测后结果储存位置
- true_test.csv：一笔正确的资料数据

## 3 Start training
- ### 印出部分训练集图片
![Training set](https://github.com/yexiaopingguo/Cat-Dog-Classification/blob/main/train_photo.png)
- ### 在CNN（Resnet）的基础上进行深度学习
    ```shell
    dog_cat_classcial.ipynb
    ```

## 4 Output prediction results
![Prediction set](https://github.com/yexiaopingguo/Cat-Dog-Classification/blob/main/pre_photo.png)

## 5 References
[1]. 猫狗图像数据来源：
https://www.kaggle.com/c/dogs-vs-cats/data
[2]. 参考kaggle竞赛奖牌获得者项目
https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification
