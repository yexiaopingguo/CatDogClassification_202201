**English** | [繁體中文](./README.zh-CN.md)（本人有點懶，翻譯功能怠工中，敬請期待....）
<h1 align="center">Cnn-Classification-Dog-Vs-Cat 貓狗辨別</h1>

**(pytorch版本) CNN Resnet18 的貓狗分類器，數據集來源於kaggle經典分類問題：貓狗大戰，基於ResNet殘差網絡及其變體網路系列，模型預測精準度高達93%（本人自建數據集正確標簽作為對比範本，判斷模型精準度）。**

## 個人心得

本項目製作於本科大三學習 [認識人工智慧AI：企業人工智慧] 課堂期間，正好碰上本人對這方面感興趣的階段，所以投入了一些熱情，選擇實作CNN網路，從入門級別的深度學習項目練手，以此作爲興趣點激勵自己學習！現階段時間有限，遂沒有自建神經網絡。參考了大量競賽獲獎選手作品，調用了已訓練的常用網絡進行深度學習，以後有時間一定會自己動手，實戰一遍殘差網絡的構建。

## 經典網絡Resnet介紹

![圖1-Resnet](https://github.com/yexiaopingguo/Cat-Dog-Classification/blob/main/photo/resnet.png)

ResNet是何凱明等人在2015年提出的模型，獲得了CVPR最佳論文獎。在ImageNet比賽classification任務上獲得第一名，因為它“簡單與實用”並存，在檢測、分割、識別等領域裡得到廣泛的應用。它使用了一種連接方式叫做“shortcut connection”，顧名思義就是“抄近道”的意思。

<img src="https://github.com/yexiaopingguo/Cat-Dog-Classification/blob/main/photo/classical_deep.png" width="300"/>

在計算機視覺裡，網絡的深度是實現網絡好的效果的重要因素，有著至關重要的影響。層數深的網絡可以提取出圖片的低層、中層和高層特徵，輸入特徵的“等級”隨增網絡深度的加深而變高。然而繼續堆疊更多層會帶來很多問題：第一個問題就是梯度爆炸 / 消失（vanishing / exploding gradients），這可以通過BN和更好的網絡初始化解決；第二個問題就是退化（degradation）問題，即當網絡層數多得飽和了，加更多層進去會導致優化困難，且訓練誤差和預測誤差更大了，網路開始退化（當網路達到一定深度，增加網路層數會導致更大的誤差），如上圖所示：

## 殘差結構

<img src="https://github.com/yexiaopingguo/Cat-Dog-Classification/blob/main/photo/shortcut.png" width="300"/>

在深度學習中，我們希望有更好性能的網絡，網絡不退化并不是我們的目的。Resnet對每層的輸入做一個reference（X）, 學習形成殘差函數， 不學習一些沒有reference（X）的函數，這種殘差函數更容易優化，能使網絡層數大大加深。

請參考上面的圖，假設輸入為 x ，有兩層全連接層學習到的映射為 H(x) ，也就是說這兩層可以漸進（asymptotically）擬合 H(x)。

假設 H(x) 與 x 維度相同，那麼擬合 H(x) 與擬合殘差函數 H(x)-x 等價，令殘差函數 F(x) = H(x) - x ，則原函數變為 F(x)+x ，
於是直接在原網絡的基礎上加上一個跨層連接，這裡的跨層連接也很簡單，就是 將 x 的恆等映射（學習的殘差函數是F(x) = H(x) - x, 這裡如果F(x) = 0, 那麼就是所謂的恆等映射 H(x) = x ）傳遞過去。

本質也就是不改變目標函數 H(x) ，將網絡結構拆成兩個分支，一個分支是殘差映射 F(x) ，另一個分支是恆等映射 x ，於是網絡僅需學習殘差映射 F(x) 即可。

**我自己總結一下**，Resnet用簡單的話來説，就是他會同時對比捲積之後和同等映射的結果，如果判斷經過捲積之後得到的是不好的結果，他就會把捲積層的參數設為0，原封不動的跳過這層捲積的作用。

## 項目流程（實戰Resnet神經網絡——貓狗分類器）

### 1. requirement
- python3
- matplotlib
- numpy
- pytorch
- pandas
- Images

### 2. Description of files
- inputs: 包含train和test數據集，來源於kaggle平台經典分類問題（貓狗大戰）[[下載地址]](https://www.kaggle.com/c/dogs-vs-cats/data)，作為入門學習，為了讓電腦處理的快一點，我把數據集數量縮小了十倍；

- dog_cat_classcial.ipynb：<font color=red>主文件</font>，Github支持在綫預覽

- ckpt_resnet18_catdog.pth：基於Resnet18的預測模型

- preds_resnet18.csv：預測後結果的儲存文件

- true_test.csv：關於測試集的正確預測文件

##### 訓練集樣本圖片
![Training set](https://github.com/yexiaopingguo/Cat-Dog-Classification/blob/main/photo/train_photo.png)

### 3 Start training
- ##### 使用GPU在Resnet18殘差網絡的基礎上進行深度學習
    ```shell
  # download the pretrained model
  import torchvision.models as models
  model = models.resnet18(pretrained = True)
  model

  # switch device to gpu if available
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ```
**更多内容請運行主文件dog_cat_classcial.ipynb主文件**

### 4 Output prediction results
![Prediction set](https://github.com/yexiaopingguo/Cat-Dog-Classification/blob/main/photo/pre_photo.png)
**模型預測精準度高達百分之九十三**

### 5 References
- [1]. <Deep Residual Learning for Image Recognition>Kaiming He,Xiangyu Zhang,Shaoqing Ren,Jian Sun
https://arxiv.org/pdf/1512.03385.pdf
- [2]. 貓狗圖片數據來源：
https://www.kaggle.com/c/dogs-vs-cats/data
- [3].參考kaggle競賽銅牌獲得者項目
https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification
- [4].Resnet殘差網絡介紹:

https://blog.csdn.net/qq_41760767/article/details/97917419

https://www.bilibili.com/video/BV1CT4y1F7KU?share_source=copy_web