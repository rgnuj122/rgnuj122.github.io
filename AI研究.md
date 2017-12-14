# AI研究
[TOC]

RNN
https://www.youtube.com/watch?v=EEtf4kNsk7Q
https://kknews.cc/zh-tw/news/6jnmq3.html
https://brohrer.mcknote.com/zh-Hant/how_machine_learning_works/how_rnns_lstm_work.html

# opensource
## Optimized Deep Learning Frameworks on Mobile Devices
-Tencent NCNN, 300 ~ 500k lib https://github.com/Tencent/ncnn
-Baidu Mobile-Deep-Learning, 340k lib https://github.com/baidu/mobile-deep-learning

# environment

## anacoda & jupiter notebook


初次下載
https://www.anaconda.com/download/#windows
![](https://i.imgur.com/0rWbntq.png)

在cmd 打 python(windows 打 py即可)
![](https://i.imgur.com/voPw9S7.png)
![](https://i.imgur.com/ZEGtz6U.png)


打 jupyter notebook
![](https://i.imgur.com/wj9Zmqg.png)
![](https://i.imgur.com/C6AknKI.png)

new一個新的python3
![](https://i.imgur.com/c0FftEe.png)
點一下檔名改名字
![](https://i.imgur.com/1pRbChp.png)
輸入後ctrl+enter執行

linear-algebra 線性代數in python練習:
筆記在紙本筆記本
https://refactored.ai/user/rf1323/notebooks/courses/linear-algebra.ipynb?track=5&type=course


## Q&A
### jupyter notebook在cmd叫不出來
前置作業
1.安裝python

2.安裝pip 
下載get-pip.py: https://pip.pypa.io/en/stable/installing/ 
![](https://i.imgur.com/NqF580j.png)

設置pip system path([參考](https://stackoverflow.com/questions/23708898/pip-is-not-recognized-as-an-internal-or-external-command))
```
D:\google drive\AI\python>setx PATH "%PATH%;C:\ProgramData\Anaconda3\Lib\site-packages"           SUCCESS: Specified value was saved.  
```
![](https://i.imgur.com/H0IbWgj.png)
無效

3.於cmd下(最高權限)輸入以下字串(不要懷疑，雙引號和括號都是必要的)
pip install "ipython[notebook]"

4.於cmd下啟動的指令如下(會在cmd目前路徑下開啟這個服務)
ipython notebook
(未來請利用下列文字啟動)
jupyter notebook

5.要離開，只要按ctrl+c，就可以離開。
(以上方法無效，從start開啟jupyter)

### 20171201 目前進度


http://www.numpy.org/
https://www.udemy.com/python-handon/learn/v4/t/lecture/8055234?start=0
![](https://i.imgur.com/h736NA1.png)

試著看AI讀書會的作業:
![](https://i.imgur.com/vgbleg3.png)

    1.最新Assignment如下
    Reading Assignment	下次讀書會日期11/29 17:00-19:00

    Bias & Variance
    http://www.cedar.buffalo.edu/~srihari/CSE574/Chap3/3.3-Bias-Variance.pdf
    Optional – 更深入的Regression Algorithm探討Ridge Regression, Lasso Regression
    http://www.math.umd.edu/~rvbalan/TEACHING/AMSC663Fall2011/PROJECTS/P6/slidesMikePekala.pdf
    http://statweb.stanford.edu/~tibs/sta305files/Rudyregularization.pdf
    Cross-Validation
    https://www.quantstart.com/articles/Using-Cross-Validation-to-Optimise-a-Machine-Learning-Method-The-Regression-Setting
    11/29 17:00-19:00
    
keras
Guiding principles
https://keras.io/
Keras安装和配置指南(Linux)
https://keras-cn.readthedocs.io/en/latest/for_beginners/keras_linux/
![](https://i.imgur.com/aT16LSL.png)

https://read01.com/zh-tw/AM4BQy.html#.WiCvwUqWaUk

![](https://i.imgur.com/heUUiEF.png)

下載duet display
https://www.duetdisplay.com/

介紹
https://www.minwt.com/mac/17569.html

>'pip' is not recognized as an internal or external command
https://stackoverflow.com/questions/23708898/pip-is-not-recognized-as-an-internal-or-external-command

沒效

https://pip.pypa.io/en/stable/installing/

# python複習
移至新頁面
https://hackmd.io/CYUwDAHAxiDsDMBaArDATIgLAI2MRAnGAcojLCCMrJiAIzyxA===?both#random

# LeNet實作

## 教材
http://hemingwang.blogspot.tw/2017/04/lenet.html

marcel說很簡單，好吧時做完要寫報告，快點寫完才能給他確認

### HW0001：

還沒有 GitHub 帳號的，請註冊一個，用來繳交作業。 註冊好後將您的 GitHub 首頁網址回覆在FB本貼文下方即可。

https://www.facebook.com/groups/pythontw/permalink/10156267862838438/?pnref=story

### HW0002：

◎ 基本題 
1. 產生五個亂數，並將其輸出。
2. 產生N個介於-1與1之間的亂數，計算其平均值與標準差並輸出，每個亂數的值則不用輸出。N=10**1, 10**2, 10**3, 10**4, 10**5。

◎ 進階題

3. 做基本題2時，一併輸出產生每N個亂數前後的系統時間，並計算所需的時間。 
4. 自己寫一個亂數產生器。

p.s.

1. 進階題可以不做。
2. 作業要有檔頭。
3. 作業要有註解。原則上英文優於繁體中文優於簡體中文優於不寫註解。
4. 作業答案請以註解方式呈現在程式碼下方。
5. 作業的連結請回覆至FB本貼文下方。 

https://www.facebook.com/groups/pythontw/permalink/10156276123488438/

sol:
```
#!/usr/bin/env python3

# HW0002 Random Number

import random
import numpy as np
import time

# Basic
# 1. Generate 5 random numbers and print

print([random.random() for i in range(5)])

# 2. Generate N random numbers within -1 to 1 and calculate mean and stddev and
# print. Random numbers shouldn't be printed. N = 10^1, 10^2, 10^3, 10^4, 10^5

# Advanced
# 3. From 2, calculate time when generating N random numbers

def rand(N):
    # 2 * [0, 1) - 1 -> [-1, 1)
    return 2 * np.random.random_sample(N) - 1

N = [10 ** i for i in range(1, 6)]
for i in N:
    start_time = time.time()
    random_number = rand(i)
    finish_time = time.time()
    print("Avg. {}".format(random_number.mean()))
    print("Stddev. {}".format(random_number.std()))
    print("Time {:6f}s".format(finish_time - start_time))

# 4. Self-made Random Number Generator

# MT19937
# https://en.wikipedia.org/wiki/Mersenne_Twister
# An Python Implementation followed by Pseudocode on Wikipedia

class RNG:
    def __init__(self, seed=0):
        self.index = 0
        self.mt = [0] * 624
        self.mt[0] = seed
        for i in range(1, 624):
            self.mt[i] = 0xFFFFFFFF & (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) +
                    i)

    def extract_number(self):
        if self.index == 0:
            self.generate_numbers()

        y = self.mt[self.index]
        y = y ^ (y >> 11)
        y = y ^ ((y << 7) & 2636928640)
        y = y ^ ((y << 15) & 4022730752)
        y = y ^ (y >> 18)

        self.index = (self.index + 1) % 624
        return y
    
    def generate_numbers(self):
        for i in range(0, 624):
            y = (self.mt[i] & 0x80000000) + (self.mt[(i + 1) % 624] & 0x7fffffff)
            self.mt[i] = self.mt[(i + 397) % 624] ^ (y >> 1)
            if y % 2 != 0:
                self.mt[i] = self.mt[i] ^ 2567483615

    def random(self):
        return self.extract_number() / 0xFFFFFFFF

rng = RNG()
print(rng.random())

# Result
'''
[0.27209326996202254, 0.05146406301581197, 0.3601347480734888, 0.5186212302047445, 0.20193117345395162]
Avg. 0.24975908156541884
Stddev. 0.591702924832753
Time 0.000051s
Avg. -0.020429980808031826
Stddev. 0.5752739352263962
Time 0.000018s
Avg. -0.0030879321112153983
Stddev. 0.569656121602261
Time 0.000024s
Avg. -0.005963881535714546
Stddev. 0.577595743356434
Time 0.000215s
Avg. -0.00038178791171763137
Stddev. 0.5768253033763888
Time 0.002251s
'''
```
Exponentiation
All that math can be done on a calculator, so why use Python? Because you can combine math with other data types (e.g. booleans) and commands to create useful programs. Calculators just stick to numbers.

![](https://i.imgur.com/7nQBn8o.png)




接下來繼續寫


# 李弘毅ML

課程清單
https://www.youtube.com/watch?v=Ky1ku1miDow&index=19&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49

![](https://i.imgur.com/gC5goCG.png)

![](https://i.imgur.com/1WA8G2Q.png)

![](https://i.imgur.com/alQzkZ8.png)

## BP演算法教學

http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/DNN%20backprop.ecm.mp4/index.html

如入寶山:
![](https://i.imgur.com/13zebk9.png)

http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLSD15_2.html


Keras : https://keras.io/

## HW1

意外找到連結
https://ntumlta.github.io/2017fall-ml-hw1/

#### 12/4進度 (10/18)
https://www.codecademy.com/courses/learn-python/lessons/functions/exercises/universal-imports?action=lesson_resume&link_content_target=

#### 12/5
what is SFCS?
http://192.192.149.24/iaeb/article.asp?aid=42

https://www.codecademy.com/courses/learn-python/lessons/python-lists-and-dictionaries/exercises/this-next-part-is-key?action=lesson_resume&link_content_target=interstitial_undefined

#### 12/6
https://www.codecademy.com/courses/learn-python/lessons/practice-makes-perfect/exercises/digitsum?action=lesson_resume&link_content_target=interstitial_undefined

patient...!

#### 12/7

寫完一堆測驗後website出問題?
![](https://i.imgur.com/9KW1PiH.png)

#### 12/8

李弘毅的Lecture 好聽
https://www.youtube.com/watch?v=FrKWiRv254g

Deep Neural Networks are Easily Fooled
https://www.youtube.com/watch?v=M2IebCN9Ht4

sigmoid  example
http://terrence.logdown.com/posts/1132631-neural-networks-with-backpropagation-one-notes

http://hemingwang.blogspot.tw/2017/06/deep-learningann.html

> 古時候 Deep Learning 叫 ANN，又叫 MLP。簡略的說法請多包涵... 
總之，現在開始儘量不講電腦、數學、跟信號處理。
類神經網路，聽起來很玄，其實，大部分的人都活在某個 ANN 裡面，
你的公司，就是一個 ANN。
你底下的工程師向你報告。你向你的主管報告，研發副總跟業務副總向總經理報告。
大家都知道，真正在做事的，只有小工程師...
工程師，就是輸入層。主管，就是隱藏層。總經理，就是輸出層。
每個人的意見，都以不同的權重被他的主管採納，總經理最後做出決策。業績不好，被董事會跟股東釘爆的話，就會回去釘副總（降低某些副總的權重）。鐵鎚釘釘子、釘子釘木板，一層一層釘下去，這個就是 BP。


> 關於 Deep Learning 的理由是，Python 可以接到 Keras （基於TensorFlow，Theano與CNTK的高階神經網路API）。也可以接到 Caffe。知道這些，應該就夠你下定決心趕快去學了！
> 


Deep Learning 101 (深度學習101)
雪豹科技的教學/交流社團
http://mit.twman.org/TonTon-Hsien-De-Huang/research/deeplearning



marcel wang網誌 基本上概念大概抓到了，可以寫報告了

李弘毅的youtube影片幫助我很多
https://www.youtube.com/watch?v=X7PH3NuYW0Q
目前從L10看到L13
L10好理解，後面比較偏細項

BP演算法
定義:反向傳播（英語：Backpropagation，縮寫為BP）是「誤差反向傳播」的簡稱，是一種與最優化方法（如梯度下降法）結合使用的，用來訓練人工神經網絡的常見方法。該方法計算對網絡中所有權重計算損失函數的梯度。這個梯度會反饋給最優化方法，用來更新權值以最小化損失函數。

學習筆記
https://ifun01.com/NWVZOFJ.html

numpy的doc
https://docs.scipy.org/doc/numpy-dev/user/basics.html
https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.html
https://docs.scipy.org/doc/numpy-dev/user/basics.creation.html
https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.array-creation.html#from-existing-data

另一人的DL
https://ithelp.ithome.com.tw/articles/10186473
https://ithelp.ithome.com.tw/articles/10187814
https://ithelp.ithome.com.tw/articles/10187912

tensor board:
https://www.tensorflow.org/get_started/summaries_and_tensorboard

numpy dashboard 
練習numpy的好地方
https://refactored.ai/tracks/


目前spyder用到hw3跟hw2
![](https://i.imgur.com/UwAGb6X.png)

注意到右上角可以ctrl+I查語法
強


#### 12/9、10

基本上就是一直聽李弘毅教授的ML
從簡入深都聽
不過對LeNet報告沒什麼幫助...

#### 12/11

往前翻才發現cloud一開始就給蠻好的資源了:
:::warning
Machine_Learning_in_Action.pdf (已經下載到我的google drive)
[點此下載](http://www2.ift.ulaval.ca/~chaib/IFT-4102-7025/public_html/Fichiers/Machine_Learning_in_Action.pdf)

可以很快的實作出python內的ML by numpy

python速查網頁:[python tutorial](https://www.tutorialspoint.com/python/index.htm)
:::

==%pylab is a "magic function" that you can call within IPython== , or Interactive Python. By invoking it, the IPython interpreter will import matplotlib and NumPy modules such that you'll have convenient access to their functions.
https://stackoverflow.com/questions/20961287/what-is-pylab

:::danger
jupyter + keras 安裝 太重要了!!!

http://www.cc.ntu.edu.tw/chinese/epaper/0041/20170620_4105.html
:::
conda create --name py-keras python=3.5
這邊可能會有一些疑惑，如果使用conda create --name py-keras即可產生一個3.6的環境，為什麼還要特別定義3.5版本號，那是因為在現行的TensorFlow的套件，僅支援到3.5版，可以參考這邊。未來可能會相容，但在本範例寫作時，還是以3.5為主。圖五顯示相容性的問題：
![](https://i.imgur.com/rzsmmsy.png)

![](https://i.imgur.com/UAGWG96.png)
activate py-keras 
![](https://i.imgur.com/yjYAtzj.png)
conda install scipy 
![](https://i.imgur.com/9ac6rBD.png)

![](https://i.imgur.com/8gU8Lsm.png)
conda install theano
![](https://i.imgur.com/x1kuZjX.png)
conda install scikit-learn
![](https://i.imgur.com/v9MTMgw.png)
conda install matplotlib
![](https://i.imgur.com/au8YPtS.png)
conda install tensorflow keras
![](https://i.imgur.com/DT75UdV.png)

conda install jupyter 
![](https://i.imgur.com/JJmDuKg.png)
![](https://i.imgur.com/D1IHXpp.png)

jupyter notebook
![](https://i.imgur.com/HgmTPK8.png)

卡住的code:

lbls, imgs = read(dataset="training", path="../data/")
![](https://i.imgur.com/nXstYm1.png)

plot_images(pooling(imgs=imgs, num_imgs=5, type_="max"))

py轉成jupyter notebook可以用的
`%load xxx.py`
就會載入整份文件了
https://stackoverflow.com/questions/21034373/how-to-load-edit-run-save-text-files-py-into-an-ipython-notebook-cell

版本差異啦幹

解决：NameError: name 'reload' is not defined 问题

对于 Python 2.X：
```python=
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

```

对于 <= Python 3.3：
```python=
import imp
imp.reload(sys)
```
注意： 
1. Python 3 与 Python 2 有很大的区别，其中Python 3 系统默认使用的就是utf-8编码。 
2. 所以，对于使用的是Python 3 的情况，就不需要sys.setdefaultencoding("utf-8")这段代码。 
3. 最重要的是，Python 3 的 sys 库里面已经没有 setdefaultencoding() 函数了。
对于 >= Python 3.4：
```python=
import importlib
importlib.reload(sys)
```

codes:

舜龍張
https://github.com/slchangtw/mAiLab/blob/master/Lab6/mAiLab_0006.ipynb


陳詠揆
https://github.com/feiyuhug/lenet-5


https://github.com/HiCraigChen/LeNet/tree/master/LeNet


LeNet:
http://blog.csdn.net/zouxy09/article/details/8781543


BP演算法
http://hemingwang.blogspot.tw/2017/02/aiback-propagation.html

目錄:http://hemingwang.blogspot.tw/2017/04/lenet.html
1. http://hemingwang.blogspot.tw/2017/04/mailab0001github.html
2. http://hemingwang.blogspot.tw/2017/04/mailab0002random-number.html
3. http://hemingwang.blogspot.tw/2017/04/mailab0003mnist-and-lenet.html
4. http://hemingwang.blogspot.tw/2017/04/mailab0004edge-detection.html
5. http://hemingwang.blogspot.tw/2017/05/mailab0005activation-function.html
6. http://hemingwang.blogspot.tw/2017/05/mailab0006pooling.html
7. http://hemingwang.blogspot.tw/2017/05/mailab0007radial-basis-function-rbf.html
8. http://hemingwang.blogspot.tw/2017/05/mailab0008lenet-5.html

这本“书”主要是《Learning IPython for Interactive Computing and Data Visualization》一书的读书笔记。
https://andersc.gitbooks.io/ipython-interactive-computing-visualization/content/4.1.figures_with_matplotlib.html

tensorflow: Mnist for ML beginner
https://www.tensorflow.org/get_started/mnist/beginners
softmax example:
https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_softmax.py

放在user資料夾，有個ML_lecture 李弘毅
![](https://i.imgur.com/USFAJBa.png)


![](https://i.imgur.com/IHGt8WR.png)

我安裝了scipy/theano/scikit/matplotlib/tensorflow keras / jupyter notebook

DEMO: http://yann.lecun.com/exdb/lenet/weirdos.html

http://deeplearning.net/tutorial/lenet.html
http://www.jianshu.com/p/7975f179ec49
Convolutional Neural Net 筆記
http://darren1231.pixnet.net/blog/post/336760136-convolutional-neural-net-%E7%AD%86%E8%A8%98

深度學習(2)--使用Tensorflow實作卷積神經網路(Convolutional neural network，CNN)
http://arbu00.blogspot.tw/2017/03/2-tensorflowconvolutional-neural.html

#### 12/13進度

azure跑不起來阿...orz
https://gallery.cortanaintelligence.com/Project/Simple-Linear-Regression

![](https://i.imgur.com/cmGhPOl.png)

https://ifun01.com/NWVZOFJ.html

http://neuralnetworksanddeeplearning.com/chap1.html

找到了!!!
https://www.tenlong.com.tw/products/9789864342167

![](https://i.imgur.com/JhaYUjB.png)

台中圖書館!!!什麼鬼都有!!!愛死你了
已預約

![](https://i.imgur.com/EJ9uxwy.png)

![](https://i.imgur.com/pkN7LyL.png)
https://www.facebook.com/groups/2027602154187130/

http://pytorch.org/