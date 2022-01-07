**English** | [繁體中文](./README.zh-CN.md)（翻譯功能還在測試中...敬請期待）
<h1 align="center">Cnn-Classification-Dog-Vs-Cat 貓狗辨別</h1>

**(pytorch版本) CNN Resnet18 的貓狗分類器，數據集來源於kaggle經典分類問題：貓狗大戰，基於ResNet殘差網絡及其變體網路系列，模型預測精準度高達93%（本人自建正確數據集作為對比範本，判斷模型精準度）。**

# 個人心得

本項目製作於本科大三學習 [認識人工智慧AI：企業人工智慧] 課堂期間，正好遇上本人對這方面感興趣的階段，所以選擇了入門的深度學習項目練手，希望做為興趣點激勵自己學習！距離DDL只剩幾天了，現階段時間和精力有限，遂沒有自建神經網絡！參考了大量競賽獲獎選手作品，調用了已訓練的常用網絡進行深度學習，以後有時間一定會自己動手，補上殘差網絡的構建。

# 經典網絡Resnet介紹

![Resnet](https://github.com/yexiaopingguo/Cat-Dog-Classification/blob/main/resnet.png)

# 項目簡介

## 1 requirement
- python3
- matplotlib
- numpy
- pytorch
- pandas
- Images

## 2 Description of files
- inputs: 包含train和test數據集，來源於kaggle平台經典分類問題（貓狗大戰）[[下載地址]](https://www.kaggle.com/c/dogs-vs-cats/data)，作為入門學習，為了讓電腦處理的快一點，我把數據集數量縮小了十倍；

- dog_cat_classcial.ipynb：<font color=red>主文件</font>，Github支持在綫預覽

- ckpt_resnet18_catdog.pth：基於Resnet18的預測模型

- preds_resnet18.csv：預測後結果的儲存文件

- true_test.csv：關於測試集的正確預測文件

#### 訓練集樣本圖片
![Training set](https://github.com/yexiaopingguo/Cat-Dog-Classification/blob/main/train_photo.png)

## 3 Start training（請運行dog_cat_classcial.ipynb文件）
- ### Resnet18殘差網絡的基礎上進行深度學習
    ```shell
  # download the pretrained model
  import torchvision.models as models
  model = models.resnet18(pretrained = True)
  model

  # switch device to gpu if available
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ```

## 4 Output prediction results
![Prediction set](https://github.com/yexiaopingguo/Cat-Dog-Classification/blob/main/pre_photo.png)
**精準度高達百分之九十三**

## 5 References
- [1]. 貓狗圖片數據來源：
https://www.kaggle.com/c/dogs-vs-cats/data
- [2].參考kaggle競賽銅牌獲得者項目
https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification
- [3].Resnet殘差網絡介紹
https://blog.csdn.net/qq_41760767/article/details/97917419?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164153307016780274186801%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=164153307016780274186801&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2