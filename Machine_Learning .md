# 機器學習 Ｍachine Learning

# 1.Introduction

### Introduce

define a set of function(model) -> goodness of function -> pick the best function

### Learning map

下圖中，同樣的顏色指的是同一個類型的事情

藍色方塊指的是scenario，即學習的情境。通常學習的情境是我們沒有辦法控制的，比如做Reinforcement Learning是因為我們沒有data、沒有辦法來做supervised Learning的情況下才去做的。如果有data，Supervised Learning當然比Reinforcement Learning要好；因此手上有什麼樣的data，就決定你使用什麼樣的scenario

紅色方塊指的是task，即要解決的問題。你要解的問題，隨著你要找的function的output的不同，有輸出scalar的regression、有輸出options的classification、有輸出structured object的structured Learning...

綠色的方塊指的是model，即用來解決問題的模型(function set)。在這些task裡面有不同的model，也就是說，同樣的task，我們可以用不同的方法來解它，比如linear model、Non-linear model(deep Learning、SVM、decision tree、K-NN... )

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_8.29.50.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_8.29.50.png)

## Supervised Learning(監督學習)

supervised learning 需要大量的training data，這些training data告訴我們說，一個我們要找的function，它的input和output之間有什麼樣的關係

而這種function的output，通常被叫做label(標籤)，也就是說，我們要使用supervised learning這樣一種技術，我們需要告訴機器，function的input和output分別是什麼，而這種output通常是通過人工的方式標註出來的，因此稱為人工標註的label，它的缺點是需要大量的人工effort

## Regression(迴歸)

regression是machine learning的一個task，特點是通過regression找到的function，它的輸出是一個scalar數值

例如：PM2.5的預測，給machine的training data是過去的PM2.5資料，而輸出的是對未來PM2.5的預測**數值**，這就是一個典型的regression的問題

## Classification(分類)

regression和classification的區別是，我們要機器輸出的東西的類型是不一樣的，在regression裡機器輸出的是scalar，而classification又分為兩類：

### Binary Classification(二元分類)

在binary classification裡，我們要機器輸出的是yes or no，是或否

比如G-mail的spam filtering(垃圾郵件過濾器)，輸入是郵件，輸出是該郵件是否是垃圾郵件

### Multi-class classification(多元分類)

在multi-class classification裡，機器要做的是選擇題，等於給他數個選項，每一個選項就是一個類別，它要從數個類別裡面選擇正確的類別

例如：document classification(新聞文章分類)，輸入是一則新聞，輸出是這個新聞屬於哪一個類別(選項)

### model(function set) 選擇模型

在解任務的過程中，第一步是要選一個function的set，選不同的function set，會得到不同的結果；而選不同的function set就是選不同的model，model又分為很多種：

- Linear Model(線性模型)：最簡單的模型
- Non-linear Model(非線性模型)：最常用的模型，包括：
    - **deep learning**
    - **SVM**
    - **decision tree**
    - **K-NN**

## Semi-supervised Learning(半監督學習)

舉例：如果想要做一個區分貓和狗的function

手頭上有少量的labeled data，它們標註了圖片上哪隻是貓哪隻是狗；同時又有大量的unlabeled data，它們僅僅只有貓和狗的圖片，但沒有標註去告訴機器哪隻是貓哪隻是狗

在Semi-supervised Learning的技術裡面，這些沒有labeled的data，對機器學習也是有幫助的

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_8.57.58.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_8.57.58.png)

### Transfer Learning(遷移學習)

假設一樣我們要做貓和狗的分類問題

我們也一樣只有少量的有labeled的data；但是我們現在有大量的不相干的data(不是貓和狗的圖片，而是一些其他不相干的圖片)，在這些大量的data裡面，它可能有label也可能沒有label

Transfer Learning要解決的問題是，這一堆不相干的data可以對結果帶來什麼樣的幫助

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_9.02.11.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_9.02.11.png)

## Unsupervised Learning(無監督學習)

區別於supervised learning，unsupervised learning希望機器學到無師自通，在完全沒有任何label的情況下，機器到底能學到什麼樣的知識

舉例來說，如果我們給機器看大量的文章，機器看過大量的文章之後，它到底能夠學到什麼事情？它能不能學會每個詞彙的意思？

學會每個詞彙的意思可以理解為：我們要找一個function，然後把一個詞彙丟進去，機器要輸出告訴你說這個詞彙是什麼意思，也許他用一個向量來表示這個詞彙的不同的特性，不同的attribute

又比如，我們帶機器去逛動物園，給他看大量的動物的圖片，對於unsupervised learning來說，我們的data中只有給function的輸入的大量圖片，沒有任何的輸出標註；在這種情況下，機器該怎麼學會根據testing data的輸入來自己生成新的圖片？

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_9.11.42.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_9.11.42.png)

## Structured Learning(結構化學習)

在Structured Learning裡，我們要機器輸出的是，一個有結構性的東西

在分類的問題中，機器輸出的只是一個選項；在structured類的problem裡面，機器要輸出的是一個複雜的物件

舉例來說，在語音識別的情境下，機器的輸入是一個聲音信號，輸出是一個句子；句子是由許多詞彙拼湊而成，它是一個有結構性的object

或者說機器翻譯、人臉識別(標出不同的人的名稱)

比如**GAN**也是Structured Learning的一種方法

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_9.16.10.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_9.16.10.png)

## Reinforcement Learning(強化學習)

**Supervised Learning：我們會告訴機器正確的答案是什麼 ，其特點是learning from teacher**

例如：訓練一個聊天機器人，告訴他如果使用者說了“Hello”，你就說“Hi”；如果使用者說了“Bye bye”，你就說“Good bye”；就好像有一個家教在它的旁邊手把手地教他每一件事情

**Reinforcement Learning**：我們沒有告訴機器正確的答案是什麼，機器最終得到的只有一個分數，就是它做的好還是不好，但他不知道自己到底哪裡做的不好，他也沒有正確的答案；很像真實社會中的學習，你沒有一個正確的答案，你只知道自己是做得好還是不好。其特點是**Learning from critics**

例如：訓練一個聊天機器人，讓它跟客人直接對話；如果客人勃然大怒把電話掛掉了，那機器就學到一件事情，剛才做錯了，它不知道自己哪裡做錯了，必須自己回去反省檢討到底要如何改進，比如一開始不應該打招呼嗎？還是中間不能罵髒話之類的

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_9.20.59.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_9.20.59.png)

再拿下棋這件事舉例，supervised Learning是說看到眼前這個棋盤，告訴機器下一步要走什麼位置；而reinforcement Learning是說讓機器和對手互弈，下了好幾手之後贏了，機器就知道這一局棋下的不錯，但是到底哪一步是贏的關鍵，機器是不知道的，他只知道自己是贏了還是輸了

其實Alpha Go是用Supervised Learning+Reinforcement Learning的方式去學習的，機器先是從棋譜學習，有棋譜就可以做supervised的學習；之後再做Reinforcement Learning，機器的對手是另外一台機器，Alpha Go就是和自己下棋，然後不斷的進步

# 2.Regression：Case Study

## 問題的導入：預測寶可夢的CP值

Estimating the Combat Power(CP) of a pokemon after evolution

我們期望根據已有的寶可夢進化前後的資訊，來預測某隻寶可夢進化後的cp值的大小

## 確定Senario、Task和Model

## Senario

首先根據已有的data來確定Senario，我們擁有寶可夢進化前後cp值的這樣一筆數據，input是進化前的寶可夢(包括它的各種屬性)，output是進化後的寶可夢的cp值；因此我們的data是labeled，使用的Senario是**Supervised Learning**

## Task

然後根據我們想要function的輸出類型來確定Task，我們預期得到的是寶可夢進化後的cp值，是一個scalar，因此使用的Task是**Regression**

## Model

關於Model，選擇很多，這裡採用的是**Non-linear Model**

## 設定具體參數

$X$： 表示一隻寶可夢，用下標表示該寶可夢的某種屬性

$X_{cp}$：表示該寶可夢進化前的cp值

$X_s$： 表示該寶可夢是屬於哪一種物種，比如妙瓜種子、皮卡丘...

$X_{hp}$：表示該寶可夢的hp值即生命值是多少

$X_w$： 代表該寶可夢的重重量

$X_h$： 代表該寶可夢的高度

$f()$： 表示我們要找的function

$y$： 表示function的output，即寶可夢進化後的cp值，是一個scalar

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_9.55.12.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_9.55.12.png)

## Regression的具體過程

### 回顧一下machine Learning的三個步驟：

- 定義一個model即function set
- 定義一個goodness of function損失函數去評估該function的好壞
- 找一個最好的function

## Step1：Model (function set)

如何選擇一個function的模型呢？畢竟只有確定了模型才能調整參數。這裡沒有明確的思路，只能憑經驗一種一種去試

## Linear Model 線性模型

$y=b+w \cdot X_{cp}$

$y$代表進化後的cp值，$X_{cp}$代表進化前的cp值， $w$和$b$代表未知參數，可以是任何數值

根據不同的$w$和$b$，可以確定不同的無窮無盡的function，而 $y=b+w \cdot X_{cp}$ 這個式子就叫做model，是以上這些具體化的function的集合，即function set

實際上這是一種**Linear Model**，但只考慮了寶可夢進化前的cp值，因而我們可以將其擴展為：

$y=b+ \sum w_ix_i$

$x_{i}$： an attribute of input X ( $x_{i}$ is also called **feature**，即特徵值)

$w_{i}$：weight of $x_{i}$

$b$： bias

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_10.12.15.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_10.12.15.png)

## Step2：Goodness of Function

## 參數說明

$x^i$：用上標來表示一個完整的object的編號，$x^{i}$表示第i只寶可夢(下標表示該object中的component)

$\widehat{y}^i$：用$\widehat{y}$表示一個實際觀察到的object輸出，上標為i表示是第i個object

註：由於Regression的輸出值是scalar，因此$\widehat{y}$裡面並沒有component，只是一個簡單的數值；但是未來如果考慮Structured Learning的時候，我們output的object可能是有structured的，所以我們還是會需要用上標下標來表示一個完整的output的object和它包含的component

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_10.18.28.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_10.18.28.png)

為了衡量function set中的某個function的好壞，我們需要一個評估函數，即 loss function 損失函數

$L(f)=L(w,b)$

input：a function；

output：how bad/good it is

由於 $f:y=b+w \cdot x_{cp}$，即`f`是由`b`和`w`決定的，因此 **input f** 就等價於 **input** 這個 **f** 裡的 **b** 和 **w**，因此 **Loss function** 實際上是在衡量一組參數的好壞

之前提到的model是由我們自主選擇的，這裡的loss function也是，最常用的方法就是採用類似於方差和的形式來衡量參數的好壞，即預測值與真值差的平方和；這裡真正的數值減估計數值的平方，叫做估計誤差，Estimation error，將10個估計誤差合起來就是loss function

$L(f)=L(w,b)=\sum_{n=1}^{10}(\widehat{y}^n-(b+w \cdot {x}^n_{cp})) ^2$

如果 $L(f)$ 越大，說明該function表現得越不好； $L(f)$ 越小，說明該function表現得越好

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_10.48.30.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_10.48.30.png)

下圖中是loss function的可視化，該圖中的每一個點都代表一組`(w,b)`，也就是對應著一個`function`；而該點的顏色對應著的loss function的結果`L(w,b)`，它表示該點對應function的表現有多糟糕，顏色越偏紅色代表Loss的數值越大，這個function的表現越不好，越偏藍色代表Loss的數值越小，這個function的表現越好

比如圖中用紅色箭頭標註的點就代表了b=-180 , w=-2對應的function，即 $y=-180-2 \cdot x_{cp}$，該點所在的顏色偏向於紅色區域，因此這個function的loss比較大，表現並不好

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_10.42.07.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_10.42.07.png)

我們已經確定了loss function，他可以衡量我們的model裡面每一個function的好壞，接下來我們要做的事情就是，從這個function set裡面，挑選一個最好的function

挑選最好的function這一件事情，寫成formulation/equation的樣子如下：

$f^*={arg} \underset{f}{min} L(f)$，或者是

$w^,b^={arg}\ \underset{w,b}{min} L(w,b)={arg}\ \underset{w,b}{min} \sum\limits^ {10}{n=1}(\widehat{y}^n-(b+w \cdot x^n{cp}))^2$

也就是那個使 $L(f)=L(w,b)=Loss$ 最小的 $f$ 或 $(w,b)$，就是我們要找的 $f^*$*或* $(w^, b^*)$ (有點像最大似然估計maximum likelihood estimation)

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_10.58.01.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_10.58.01.png)

利用線性代數的知識，可以解得這個closed-form solution，但這裡採用的是一種更為普遍的方法—   **gradient descent(梯度下降法)**

## Gradient Descent 梯度下降

上面的例子比較簡單，用線性代數的知識就可以解；但是對於更普遍的問題來說，**gradient descent**的厲害之處在於，只要 $L(f)$ 是可微分的， **gradient descent**都可以拿來處理這個 $f$，找到表現比較好的parameters

## 單個參數的問題

以只帶單個參數w的Loss Function $L(w)$ 為例，首先保證 $L(w)$ 是**可微**的
$w^={arg}\ \underset{w}{min} L(w)$ *我們的目標就是找到這個使Loss最小的$w^*$，實際上就是尋找切線L斜率為0的global minima最小值點(注意，存在一些local minima極小值點，其斜率也是0)

有一個暴力的方法是，窮舉所有的w值，去找到使loss最小的 $w^*$，但是這樣做是沒有效率的；而 **gradient descent**就是用來解決這個效率問題的

首先隨機選取一個初始的點 $w^0$ (當然也不一定要隨機選取，如果有辦法可以得到比較接近 $w^*$*的表現得比較好的* $w^0$*當初始點，可以有效地提高查找* $w^*$的效率)

計算 $L$ 在 $w=w^0$的位置的微分，即 $\frac{dL}{dw}|_{w=w^0}$，幾何意義就是切線的斜率

如果切線斜率是negative負的，那麼就應該使$w$變大，即往右踏一步；如果切線斜率是positive正的，那麼就應該使$w$變小，即往左踏一步，每一步的步長step size就是w的改變量

w的改變量step size的大小取決於兩件事

- 一是現在的微分值 $\frac{dL}{dw}$有多大，微分值越大代表現在在一個越陡峭的地方，那它要移動的距離就越大，反之就越小
- 二是一個常數項 $η$，被稱為 **learning rate**，即學習率，它決定了每次踏出的step size不只取決於現在的斜率，還取決於一個事先就定好的數值，如果learning rate比較大，那每踏出一步的時候，參數w更新的幅度就比較大，反之參數更新的幅度就比較小

如果learning rate設置的大一些，那機器學習的速度就會比較快；但是learning rate如果太大，可能就會跳過最合適的global minima的點

因此每次參數更新的大小是 $η \frac{dL}{dw}$，為了滿足斜率為負時$w$變大，斜率為正時$w$變小，應當使原來的 $w$減去更新的數值，即

$w^1=w^0-η \frac{dL}{dw}|_{w=w^0} \\w^2=w^1-η \frac{dL}{dw}|_{w=w^1} \\w^3=w^2-η \frac{dL}{dw}|_{w=w^2} \\... \\w^{i+1}=w^i-η \frac{dL}{dw}|_{w=w^i} \\if\ \ (\frac{dL}{dw}|_{w=w^i}==0) \ \ then \ \ stop$

此時 $w^i$ 對應的斜率為0，我們找到了一個極小值local minima，這就出現了一個問題，當微分為0 的時候，參數就會一直卡在這個點上沒有辦法再更新了，因此通過gradient descent找出來的solution其實並不是最佳解global minima

但幸運的是，在linear regression上，是沒有local minima的，因此可以使用這個方法

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_11.21.46.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-05_11.21.46.png)

## 兩個參數的問題

今天要解決的關於寶可夢的問題，是含有two parameters的問題，即 $(w^*,b^*)=arg\ \underset{w,b} {min} L(w,b)$

當然，它本質上處理單個參數的問題是一樣的

- 首先，也是隨機選取兩個初始值， $w^0$和 $b^0$
- 然後分別計算 $(w^0,b^0)$這個點上，L對w和b的偏微分，即 $\frac{\partial L}{\partial w}|{w=w^0 ,b=b^0}$ *和* $\frac{\partial L}{\partial b}|{w=w^0,b=b^0}$
- 更新參數，當迭代跳出時， $(w^i,b^i)$對應著極小值點

$w^1=w^0-η\frac{\partial L}{\partial w}|_{w=w^0,b=b^0} \ \ \ \ \ \ \ \ \ b^1=b ^0-η\frac{\partial L}{\partial b}|_{w=w^0,b=b^0} \\
    w^2=w^1-η\frac{\partial L}{\partial w}|_{w=w^1,b=b^1} \ \ \ \ \ \ \ \ \ b^2=b ^1-η\frac{\partial L}{\partial b}|_{w=w^1,b=b^1} \\
    ... \\
    w^{i+1}=w^{i}-η\frac{\partial L}{\partial w}|_{w=w^{i},b=b^{i}} \ \ \ \ \ \ \ \ \ b^{i+1}=b^{i}-η\frac{\partial L}{\partial b}|_{w=w^{i},b=b^{i} } \\
    if(\frac{\partial L}{\partial w}==0 \&\& \frac{\partial L}{\partial b}==0) \ \ \ then \ \ stop$

實際上，L 的gradient就是微積分中的梯度的概念，即

$\nabla L=\begin{bmatrix}\frac{\partial L}{\partial w} \\\frac{\partial L}{\partial b}
\end{bmatrix}_{gradient}$

可視化效果如下：(三維坐標顯示在二維圖像中，loss的值用顏色來表示)

橫坐標是b，縱坐標是w，顏色代表loss的值，越偏藍色表示loss越小，越偏紅色表示loss越大

**每次計算得到的梯度gradient，即由** $\frac{\partial L}{\partial b}和\frac{\partial L}{\partial w}$ **組成的vector向量，就是該等高線的法線方向(對應圖中紅色箭頭的反方向)；而** $(-η\frac{\partial L}{\partial b},-η\frac{\partial L}{\partial w})$ **的作用就是讓原先的** $(w^i,b^i)$ **朝著gradient的反方向即等高線法線方向前進，其中η(learning rate)的作用是每次更新的跨度(對應圖中紅色箭頭的長度)；經過多次迭代，最終gradient達到極小值點**

註：這裡兩個方向的η(learning rate)必須保持一致，這樣每次更新坐標的step size是等比例縮放的，保證坐標前進的方向始終和梯度下降的方向一致；否則坐標前進的方向將會發生偏移

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.02.30.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.02.30.png)

## Gradient Descent的缺點

gradient descent有一個令人擔心的地方，也就是我之前一直提到的，它每次迭代完畢，尋找到的梯度為0的點必然是極小值點，local minima；卻不一定是最小值點，global minima

這會造成一個問題是說，如果loss function長得比較坑坑疤疤(極小值點比較多)，而每次初始化 $w^0$的取值又是隨機的，這會造成每次gradient descent停下來的位置都可能是不同的極小值點；而且當遇到梯度比較平緩(gradient≈0)的時候，gradient descent也可能會效率低下甚至可能會卡住；也就是說通過這個方法得到的結果，是看人品的

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.06.32.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.06.32.png)

但是！在**linear regression**裡，loss function實際上是**convex**的，是一個**凸函數**，是沒有local optimal局部最優解的，他只有一個global minima，visualize出來的圖像就是從裡到外一圈一圈包圍起來的橢圓形的等高線(就像前面的等高線圖)，因此隨便選一個起始點，根據gradient descent最終找出來的，都會是同一組參數

## 回到pokemon的問題上來

### 偏微分的計算

現在我們來求具體的L對w和b的偏微分

$L(w,b)=\sum\limits_{n=1}^{10}(\widehat{y}^n-(b+w\cdot x_{cp}^n))^2 \\\frac{\partial L}{\partial w}=\sum\limits_{n=1}^{10}2(\widehat{y}^n-(b+w\cdot x_{cp}^n)) (-x_{cp}^n) \\\frac{\partial L}{\partial b}=\sum\limits_{n=1}^{10}2(\widehat{y}^n-(b+w\cdot x_{cp}^n)) (-1)$

### How's the results?

根據gradient descent，我們得到的 $y=b+w\cdot x_{cp}$ 中最好的參數是b=-188.4, w=2.7

我們需要有一套評估系統來評價我們得到的最後這個function和實際值的誤差error的大小；這裡我們將training data裡每一隻寶可夢 $i$ 進化後的實際cp值與預測值之差的絕對值叫做 $e^i$，而這些誤差之和Average Error on Training Data為 $\sum\limits_{i=1}^{10}e^i=31.9$

**What we really care about is the error on new data (testing data)**
當然我們真正關心的是generalization的case，也就是用這個model去估測新抓到的pokemon，誤差會有多少，這也就是所謂的testing data的誤差；於是又抓了10只新的pokemon，算出來的Average Error on Testing Data為 $\sum\limits_{i=1}^{10}e^i=35.0$；可見training data裡得到的誤差一般是要比testing data要小，這也符合常識

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.17.07.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.17.07.png)

我們有沒有辦法做得更好呢？這時就需要我們重新去設計model；如果仔細觀察一下上圖的data，就會發現在原先的cp值比較大和比較小的地方，預測值是相當不准的

實際上，從結果來看，最終的function可能不是一條直線，可能是稍微更複雜一點的曲線

### 考慮 $(x_{cp})^2$ 的model

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.19.23.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.19.23.png)

### 考慮 $(x_{cp})^3$ 的model

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.21.00.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.21.00.png)

### 考慮 $(x_{cp})^4$ 的model

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.23.06.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.23.06.png)

### 考慮 $(x_{cp})^5$ 的model

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.24.45.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.24.45.png)

### 5個model的對比

這5個model的training data的表現：隨著 $(x_{cp})^i$ 的高次項的增加，對應的average error會不斷地減小；實際上這件事情非常容易解釋，實際上低次的式子是高次的式子的特殊情況(令高次項 $(X_{cp})^i$對應的 $w_i$ 為0，高次式就轉化成低次式)

也就是說，在gradient descent可以找到best function的前提下(多次式為Non-linear model，存在local optimal局部最優解，gradient descent不一定能找到global minima)，function所包含的項的次數越高，越複雜，error在training data上的表現就會越來越小；但是，我們關心的不是model在training data上的error表現，而是model在testing data上的error表現

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.27.36.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.27.36.png)

在training data上，model越複雜，error就會越低；但是在testing data上，model複雜到一定程度之後，error非但不會減小，反而會暴增，在該例中，從含有 $(X_ {cp})^4$ 項的model開始往後的model，testing data上的error出現了大幅增長的現象，通常被稱為**overfitting過擬合**

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.30.44.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.30.44.png)

因此model不是越複雜越好，而是選擇一個最適合的model

## 進一步討論其他參數

### 物種 $x_s$ 的影響

之前我們的model只考慮了寶可夢進化前的cp值，這顯然是不對的，除了cp值外，還受到物種 $x_s$的影響

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.35.01.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.35.01.png)

因此我們重新設計model：

$if \ \ x_s=Pidgey: \ \ \ \ \ \ \ y=b_1+w_1\cdot x_{cp} \\if \ \ x_s=Weedle: \ \ \ \ \ \ y=b_2+w_2\cdot x_{cp} \\if \ \ x_s=Caterpie: \ \ \ \ y=b_3+w_3\cdot x_{cp} \\if \ \ x_s=Eevee: \ \ \ \ \ \ \ \ \ y=b_4+w_4\cdot x_{cp}$

也就是根據不同的物種，設計不同的linear model(這裡 $x_s=species \ of \ x$ )，那如何將上面的四個if語句合併成一個linear model呢？

這裡引入 $δ(條件式)$的概念，當條件式為true，則δ為1；當條件表達式為false，則δ為0，因此可以通過下圖的方式，將4個if語句轉化成同一個linear model

有了上面這個model以後，我們分別得到了在training data和testing data上測試的結果：

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.39.33.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.39.33.png)

### Hp值 $x_{hp}$、height值 $x_h$、weight值 $x_w$的影響

考慮所有可能有影響的參數，設計出這個最複雜的model：

$if \ \ x_s=Pidgey: \ \ \ \ y'=b_1+w_1\cdot x_{cp}+w_5\cdot(x_{cp})^2 \\if \ \ x_s=Weedle: \ \ \ y'=b_2+w_2\cdot x_{cp}+w_6\cdot(x_{cp})^2 \\if \ \ x_s=Pidgey: \ \ \ y'=b_3+w_3\cdot x_{cp}+w_7\cdot(x_{cp})^2 \\if \ \ x_s=Eevee: \ \ \ \ y'=b_4+w_4\cdot x_{cp}+w_8\cdot(x_{cp})^2 \\y=y'+w_9\cdot x_{hp}+w_{10}\cdot(x_{hp})^2+w_{11}\cdot x_h+w_{12}\cdot (x_h)^2+w_{ 13}\cdot x_w+w_{14}\cdot (x_w)^2$

算出的training error=1.9，但是，testing error=102.3！ **這麼複雜的model很大概率會發生overfitting**(overfitting實際上是我們多使用了一些input的變量或是變量的高次項使曲線跟training data擬合的更好，但不幸的是這些項並不是實際情況下被使用的，於是這個model在testing data上會表現得很糟糕)，overfitting就相當於是那個範圍更大的文氏圖，它包含了更多的函數更大的範圍，代價就是在準確度上表現得更糟糕

## regularization解決overfitting(L2正則化解決過擬合問題)

regularization可以使曲線變得更加smooth，training data上的error變大，但是 testing data上的error變小。原來的loss function只考慮了prediction的error，即

$\sum\limits_i^n(\widehat{y}^i-(b+\sum\limits_{j}w_jx_j))^2$

而regularization則是在原來的loss function的基礎上加上了一項

$\lambda\sum(w_i)^2$，就是把這個model裡面所有的 $w_i$的平方和用λ加權(其中i代表traversal n個training data，j代表traversal model的每一項)

也就是說， **我們期待參數** $w_i$ **越小甚至接近於0的function，為什麼呢?**

因為參數值接近0的function，是比較平滑的；所謂的平滑的意思是，當今天的輸入有變化的時候，output對輸入的變化是比較不敏感的

舉例來說，對 $y=b+\sum w_ix_i$ 這個model，當input變化 $\Delta x_i$，output的變化就是 $w_i\Delta x_i$，也就是說，如果 $w_i$越小越接近0的話，輸出對輸入就越不sensitive敏感，我們的function就是一個越平滑的function；說到這裡你會發現，我們之前沒有把bias——b這個參數考慮進去的原因是**bias的大小跟function的平滑程度是沒有關係的**，bias值的大小只是把function上下移動而已

**那為什麼我們喜歡比較平滑的function呢？**

如果我們有一個比較平滑的function，由於輸出對輸入是不敏感的，測試的時候，一些noises雜訊對這個平滑的function的影響就會比較小，而給我們一個比較好的結果

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.54.16.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.54.16.png)

註**：這裡的λ需要我們手動去調整以取得最好的值**

λ值越大代表考慮smooth的那個regularization那一項的影響力越大，我們找到的function就越平滑

觀察下圖可知，當我們的λ越大的時候，在training data上得到的error其實是越大的，但是這件事情是非常合理的，因為當λ越大的時候，我們就越傾向於考慮w的值而越少考慮error的大小；但是有趣的是，雖然在training data上得到的error越大，但是在testing data上得到的error可能會是比較小的

下圖中，當λ從0到100變大的時候，training error不斷變大，testing error反而不斷變小；但是當λ太大的時候(>100)，在testing data上的error就會越來越大

我們喜歡比較平滑的function，因為它對noise不那麼sensitive；但是我們又不喜歡太平滑的function，因為它就失去了對data擬合的能力；而function的平滑程度，就需要通過調整λ來決定，就像下圖中，當λ=100時，在testing data上的error最小，因此我們選擇λ=100

註：這裡的error指的是 $\frac{1}{n}\sum\limits_{i=1}^n|\widehat{y}^i-y^i|$

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.58.01.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_2.58.01.png)

## conclusion總結

### 關於pokemon的cp值預測的流程總結：

- 根據已有的data特點(labeled data，包含寶可夢及進化後的cp值)，確定使用supervised learning監督學習
- 根據output的特點(輸出的是scalar數值)，確定使用regression回歸(linear or non-linear)
- 考慮包括進化前cp值、species、hp等各方面變量屬性以及高次項的影響，我們的model可以採用這些input的一次項和二次項之和的形式，如：

 $if \ \ x_s=Pidgey: \ \ \ \ y'=b_1+w_1\cdot x_{cp}+w_5\cdot(x_{cp})^2 \\    if \ \ x_s=Weedle: \ \ \ y'=b_2+w_2\cdot x_{cp}+w_6\cdot(x_{cp})^2 \\    if \ \ x_s=Pidgey: \ \ \ y'=b_3+w_3\cdot x_{cp}+w_7\cdot(x_{cp})^2 \\    if \ \ x_s=Eevee: \ \ \ \ y'=b_4+w_4\cdot x_{cp}+w_8\cdot(x_{cp})^2 \\    y=y'+w_9\cdot x_{hp}+w_{10}\cdot(x_{hp})^2+w_{11}\cdot x_h+w_{12}\cdot (x_h)^2+w_{ 13}\cdot x_w+w_{14}\cdot (x_w)^2$

而為了保證function的平滑性，loss function應使用regularization，即

 $L=\sum\limits_{i=1}^n(\widehat{y}^iy^i)^2+\lambda\sum\limits_{ j}(w_j)^2$，注意bias——參數b對function平滑性無影響，因此不額外再次計入loss function(y的表達式裡已包含w、b)

- 利用gradient descent對regularization版本的loss function進行梯度下降迭代處理，每次迭代都減去L對該參數的微分與learning rate之積，假設所有參數合成一個vector：$[w_0,w_1,w_2,. ..,w_j,...,b]^T$，那麼每次梯度下降的表達式如下：

 $梯度:   \nabla L=    \begin{bmatrix}    \frac{\partial L}{\partial w_0} \\    \frac{\partial L}{\partial w_1} \\    \frac{\partial L}{\partial w_2} \\    ... \\    \frac{\partial L}{\partial w_j} \\    ... \\    \frac{\partial L}{\partial b}    \end{bmatrix}_{gradient}    \ \ \$    

$gradient \ descent:    \begin{bmatrix}    w'_0\\    w'_1\\    w'_2\\    ...\\    w'_j\\    ...\\    b'    \end{bmatrix}_{L=L'}    = \ \ \ \ \ \    \begin{bmatrix}    w_0\\    w_1\\    w_2\\    ...\\    w_j\\    ...\\    b    \end{bmatrix}_{L=L_0}    -\ \ \ \ \eta    \begin{bmatrix}    \frac{\partial L}{\partial w_0} \\    \frac{\partial L}{\partial w_1} \\    \frac{\partial L}{\partial w_2} \\    ... \\    \frac{\partial L}{\partial w_j} \\    ... \\    \frac{\partial L}{\partial b}    \end{bmatrix}_{L=L_0}$

當梯度穩定不變時，即 $\nabla L$為0時，gradient descent便停止，此時如果採用的model是linear的，那麼vector必然落於global minima處(凸函數)；如果採用的model是Non-linear的，vector可能會落於local minima處(此時需要採取其他辦法獲取最佳的function)

假定我們已經通過各種方法到達了global minima的地方，此時的vector：  $[w_0,w_1,w_2,...,w_j,...,b]^T$所確定的那個唯一的function就是在該λ下的最佳 $f^*$，即loss最小

這裡λ的最佳數值是需要通過我們不斷調整來獲取的，因此令λ等於0，10，100，1000，...不斷使用gradient descent或其他算法得到最佳的parameters： $[w_0,w_1 ,w_2,...,w_j,...,b]^T$，併計算出這組參數確定的function—— $f^*$對training data和testing data上的error值，直到找到那個使testing data的error最小的λ，(這裡一開始λ=0，就是沒有使用regularization時的loss function)

註：引入評價 $f^*$*的error機制，令error=* $\frac{1}{n}\sum\limits_{i=1}^n|\widehat{y}^iy^i|$*，分別計算該* $f^*$對training data和testing data(more important)的 $error(f^*)$大小

先設定λ->確定loss function->找到使loss最小的 $[w_0,w_1,w_2,...,w_j,...,b]^T$ 

->確定function->計算error->重新設定新的λ重複上述步驟

->使testing data上的error最小的λ所對應的 $[w_0,w_1,w_2,...,w_j,...,b]^T$所對應的function就是我們能夠找到的最佳的function

### 本章節總結：

- Pokémon: Original CP and species almost decide the CP after evolution
- There are probably other hidden factors
- Gradient descent
    - More theory and tips in the following lectures
- Overfitting and Regularization
- We finally get average error = 11.1 on the testing data
- How about new data? Larger error? Lower error?(larger->need validation)
- Next lecture: Where does the error come from?
    - More theory about overfitting and regularization
    - The concept of validation(用來解決new data的error高於11.1的問題)

# Regression：linear model

這裡用的是 Adagrad ，接下來的課程會再細講，這裡只是想顯示 gradient descent 實作起來沒有想像的那麼簡單，還有很多小技巧要注意

這裡採用最簡單的linear model：**y_data=b+w*x_data**

我們要用gradient descent把b和w找出來

當然這個問題有closed-form solution，這個b和w有更簡單的方法可以找出來；那我們假裝不知道這件事，我們練習用gradient descent把b和w找出來

## Demo

### 數據準備：

```python
# 假設x_data和y_data都有10筆，分別代表寶可夢進化前後的cp值
x_data=[338.,333.,328.,207.,226.,25.,179.,60.,208.,606.]
y_data=[640.,633.,619.,393.,428.,27.,193.,66.,226.,1591.]
# 這裡採用最簡單的linear model：y_data=b+w*x_data
# 我們要用gradient descent把b和w找出來
```

### 計算梯度微分的函數getGrad()

```python
# 計算梯度微分的函數getGrad()
def getGrad(b,w):
    # initial b_grad and w_grad
    b_grad=0.0
    w_grad=0.0
    for i in range(10):
        b_grad+=(-2.0)*(y_data[i]-(b+w*x_data[i]))
        w_grad+=(-2.0*x_data[i])*(y_data[i]-(b+w*x_data[i]))
    return (b_grad,w_grad)
```

### 引入需要的庫

```python
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
%matplotlib inline
import random as random
import numpy as np
import csv

```

### 準備好b、w、loss的圖像數據

```python
# 生成一組b和w的數據圖，方便給gradient descent的過程做標記
x = np.arange(-200,-100,1) # bias
y = np.arange(-5,5,0.1) # weight
Z = np.zeros((len(x),len(y))) # color
X,Y = np.meshgrid(x,y)
for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        
        # Z[j][i]存儲的是loss
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (y_data[n] - (b + w * x_data[n]))**2
        Z[j][i] = Z[j][i]/len(x_data)

```

### 規定迭代次數和learning rate，進行第一次嘗試

距離最優解還有一段距離

```python
# y_data = b + w * x_data
b = -120 # initial b
w = -4 # initial w
lr = 0.0000001 # learning rate
iteration = 100000 # 這裡直接規定了迭代次數，而不是一直運行到b_grad和w_grad都為0(事實證明這樣做不太可行)
# store initial values for plotting，我們想要最終把數據描繪在圖上，因此存儲過程數據
b_history = [b]
w_history = [w]
# iterations
for i in range(iteration):
    
    # get new b_grad and w_grad
    b_grad,w_grad=getGrad(b,w)
    
    # update b and w
    b -= lr * b_grad
    w -= lr * w_grad
    
    #store parameters for plotting
    b_history.append(b)
    w_history.append(w)
# plot the figure
plt.contourf(x,y,Z,50,alpha=0.5,cmap=plt.get_cmap('jet'))
plt.plot([-188.4],[2.67],'x',ms=12,markeredgewidth=3,color='orange')
plt.plot(b_history,w_history,'o-',ms=3,lw=1.5,color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$',fontsize=16)
plt.ylabel(r'$w$',fontsize=16)
plt.show()

```

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_4.41.15.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_4.41.15.png)

### 把learning rate增大10倍嘗試

發現經過100000次的update以後，我們的參數相比之前與最終目標更接近了，但是這裡有一個劇烈的震盪現象發生

```python
# 上圖中，gradient descent最終停止的地方裡最佳解還差很遠，
# 由於我們是規定了iteration次數的，因此原因應該是learning rate不夠大，這裡把它放大10倍
# y_data = b + w * x_data
b = -120 # initial b
w = -4 # initial w
lr = 0.000001 # learning rate 放大10倍
iteration = 100000 # 這裡直接規定了迭代次數，而不是一直運行到b_grad和w_grad都為0(事實證明這樣做不太可行)
# store initial values for plotting，我們想要最終把數據描繪在圖上，因此存儲過程數據
b_history = [b]
w_history = [w]
# iterations
for i in range(iteration):
    
    # get new b_grad and w_grad
    b_grad,w_grad=getGrad(b,w)
    
    # update b and w
    b -= lr * b_grad
    w -= lr * w_grad
    
    #store parameters for plotting
    b_history.append(b)
    w_history.append(w)
# plot the figure
plt.contourf(x,y,Z,50,alpha=0.5,cmap=plt.get_cmap('jet'))
plt.plot([-188.4],[2.67],'x',ms=12,markeredgewidth=3,color='orange')
plt.plot(b_history,w_history,'o-',ms=3,lw=1.5,color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$',fontsize=16)
plt.ylabel(r'$w$',fontsize=16)
plt.show()

```

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_4.43.18.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_4.43.18.png)

### 把learning rate再增大10倍

發現此時learning rate太大了，參數一update，就遠遠超出圖中標註的範圍了

所以我們會發現一個很嚴重的問題，如果learning rate變小一點，他距離最佳解還是會具有一段距離；但是如果learning rate放大，它就會直接超出範圍了

```python
# 上圖中，gradient descent最終停止的地方裡最佳解還是有一點遠，
# 由於我們是規定了iteration次數的，因此原因應該是learning rate還是不夠大，這裡再把它放大10倍
# y_data = b + w * x_data
b = -120 # initial b
w = -4 # initial w
lr = 0.00001 # learning rate 放大10倍
iteration = 100000 # 這裡直接規定了迭代次數，而不是一直運行到b_grad和w_grad都為0(事實證明這樣做不太可行)
# store initial values for plotting，我們想要最終把數據描繪在圖上，因此存儲過程數據
b_history = [b]
w_history = [w]
# iterations
for i in range(iteration):
    
    # get new b_grad and w_grad
    b_grad,w_grad=getGrad(b,w)
    
    # update b and w
    b -= lr * b_grad
    w -= lr * w_grad
    
    #store parameters for plotting
    b_history.append(b)
    w_history.append(w)
# plot the figure
plt.contourf(x,y,Z,50,alpha=0.5,cmap=plt.get_cmap('jet'))
plt.plot([-188.4],[2.67],'x',ms=12,markeredgewidth=3,color='orange')
plt.plot(b_history,w_history,'o-',ms=3,lw=1.5,color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$',fontsize=16)
plt.ylabel(r'$w$',fontsize=16)
plt.show()

```

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_4.45.57.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_4.45.57.png)

這個問題明明很簡單，可是只有兩個參數b和w，gradient descent搞半天都搞不定，那以後做neural network有數百萬個參數的時候，要怎麼辦呢

這個就是**一室之不治，何以天下國家為**的概念

### 解決方案：Adagrad

我們給b和w訂製化的learning rate，讓它們兩個的learning rate不一樣

```python
# 這裡給b和w不同的learning rate
# y_data = b + w * x_data
b = -120 # initial b
w = -4 # initial w
lr = 1 # learning rate 放大10倍
iteration = 100000 # 這裡直接規定了迭代次數，而不是一直運行到b_grad和w_grad都為0(事實證明這樣做不太可行)
# store initial values for plotting，我們想要最終把數據描繪在圖上，因此存儲過程數據
b_history = [b]
w_history = [w]
lr_b = 0
lr_w = 0
# iterations
for i in range(iteration):
    
    # get new b_grad and w_grad
    b_grad,w_grad=getGrad(b,w)
    
    # get the different learning rate for b and w
    lr_b = lr_b + b_grad ** 2
    lr_w = lr_w + w_grad ** 2
    
    # 這一招叫做adagrad，之後會詳加解釋
    # update b and w with new learning rate
    b -= lr / np.sqrt(lr_b) * b_grad
    w -= lr / np.sqrt(lr_w) * w_grad
    
    #store parameters for plotting
    b_history.append(b)
    w_history.append(w)
    
    # output the b w b_grad w_grad
    # print("b: "+str(b)+"\\t\\t\\tw: "+str(w)+"\\n"+"b_grad: "+str(b_grad)+"\\t\\t w_grad: "+str(w_grad)+"\\n")
    
# output the final function and its error
print("the function will be y_data="+str(b)+"+"+str(w)+"*x_data")
error=0.0
for i in range(10):
    print("error "+str(i)+" is: "+str(np.abs(y_data[i]-(b+w*x_data[i])))+" ")
    error+=np.abs(y_data[i]-(b+w*x_data[i]))
average_error=error/10
print("the average error is "+str(average_error))
# plot the figure
plt.contourf(x,y,Z,50,alpha=0.5,cmap=plt.get_cmap('jet'))
plt.plot([-188.4],[2.67],'x',ms=12,markeredgewidth=3,color='orange')
plt.plot(b_history,w_history,'o-',ms=3,lw=1.5,color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$',fontsize=16)
plt.ylabel(r'$w$',fontsize=16)
plt.show()

```

the function will be y_data=-188.3668387495323+2.6692640713379903*x_data
error 0 is: 73.84441736270833
error 1 is: 67.4980970060185
error 2 is: 68.15177664932844
error 3 is: 28.8291759825683
error 4 is: 13.113158627146447
error 5 is: 148.63523696608252
error 6 is: 96.43143001996799
error 7 is: 94.21099446925288
error 8 is: 140.84008808876973
error 9 is: 161.7928115187101
the average error is 89.33471866905532

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_4.49.53.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_4.49.53.png)

**有了新的learning rate以後，從初始值到終點，我們在100000次iteration之內就可以順利地完成了**

# Where does the error come from?

### Review

之前有提到說，不同的function set，也就是不同的model，它對應的error是不同的；越複雜的model，也許performance會越差，所以今天要討論的問題是，這個error來自什麼地方

- error due to **bias**
- error due to **variance**

了解error的來源其實是很重要的，因為我們可以針對它挑選適當的方法來improve自己的model，提高model的準確率，而不會毫無頭緒

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_4.55.14.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_4.55.14.png)

### 抽樣分佈

### $\widehat{y}$ 和 $y^*$ 真值和估計值

$\widehat{y}$ 表示那個真正的function，而 $f^*$ 表示這個 $\widehat{f}$ 的估計值estimator

就好像在打靶， $\widehat{f}$ 是靶的中心點，收集到一些data做training以後，你會得到一個你覺得最好的function即 $f^*$*，這個* $f^*$ 落在靶上的某個位置，它跟靶中心有一段距離，這段距離就是由Bias和variance決定的

bias：偏差；variance：方差-> 實際上對應著物理實驗中系統誤差和隨機誤差的概念，假設有n組數據，每一組數據都會產生一個相應的 $f^*$*，此時bias表示所有* $f^*$ 的平均落靶位置和真值靶心的距離，variance表示這些 $f^*$ 的集中程度

### 抽樣分佈的理論(概率論與數理統計)

假設獨立變量為x(這裡的x代表每次獨立地從不同的training data裡訓練找到的 $f^*$ )，那麼

總體期望值 $E(x)=u$ ；總體變異數 $Var(x)=\sigma^2$

### 用樣本平均值 $\overline{x}$ 估計總體期望值 $u$

由於我們只有有限組樣本 $Sample \ N \ points:\{x^1,x^2,...,x^N\}$，故

樣本平均值 $\overline{x}=\frac{1}{N}\sum\limits_{i=1}^{N}x^i$ ；樣本平均值的期望 $E(\overline{x})=E (\frac{1}{N}\sum\limits_{i=1}^{N}x^i)=u$ ;

樣本平均值的變異數 $Var(\overline{x})=\frac{\sigma^ 2}{N}$

**樣本平均值** $\overline{x}$ **的期望是總體期望值** $u$，也就是說 $\overline{x}$ 是按機率對稱地分佈在總體期望 $u$ 的兩側的；而 $\overline{x}$ 分佈的密集程度取決於N，即數據量的大小，如果N比較大， $\overline{x}$ 就會比較集中，如果N比較小， $\overline{x}$ 就會以 $u$ 為中心分散開來

綜合以上，樣本平均值 $\overline{x}$ 以總體期望值 $u$ 為中心對稱分佈，可以用來估計總體期望值 $u$

### 用樣本變異數 $s^2$ 估計總體變異數 $\sigma^2$

由於我們只有有限組樣本 $Sample \ N \ points:\{x^1,x^2,...,x^N\}$，故

樣本平均值 $\overline{x}=\frac{1}{N}\sum\limits_{i=1}^{N}x^i$；樣本變異數 $s^2=\frac{1}{N- 1}\sum\limits_{i=1}^N(x^i-\overline{x})^2$ ；

樣本變異數的期望 $E(s^2)=\sigma^2$ ； 樣本變異數的變異數 $Var(s^2)=\frac{2\sigma^4}{N-1}$

**樣本變異數** $s^2$ **的期望值是總體變異數** $\sigma^2$，而 $s^2$ 分佈的密集程度也取決於N

同理，樣本變異數 $s^2$ 以總體變異數 $\sigma^2$ 為中心對稱分佈，可以用來估計總體變異數 $\sigma^2$

## 回到regression的問題上來

現在我們要估計的是靶的中心 $\widehat{f}$，每次collect data訓練出來的 $f^*$ 是打在靶上的某個點；產生的error取決於：

- 多次實驗得到的 $f^*$ *的期望值* $\overline{f}$ *與靶心* $\widehat{f}$ *之間的bias——* $E(f^*)$，可以形像地理解為瞄準的位置和靶心的距離的偏差
- 多次實驗的$f^*$之間的variance——$Var(f^*)$，可以形像地理解為多次打在靶上的點的集中程度

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_5.23.04.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_5.23.04.png)

說到這裡，可能會產生一個疑惑：我們之前不就只做了一次實驗嗎？我們就collect了十筆data，然後training出來了一個 $f^*$*，然後就結束了。那怎麼找很多個* $f^*$ 呢？怎麼知道它的bias和variance有多大呢？

## $f^*$取決於model的複雜程度以及data的數量

假設這裡有多個平行宇宙，每個空間裡都在用10只寶可夢的data去找 $f^*$於不同宇宙中寶可夢的data是不同的，因此即使使用的是同一個model，最終獲得的 **$f^*$ 都會是不同的

於是我們做100次相同的實驗，把這100次實驗找出來的100條 $f^*$ 的分佈畫出來

## $f^*$的variance取決於model的複雜程度和data的數量

$f^*$*的variance是由model決定的，一個簡單的model在不同的training data下可以獲得比較穩定分佈的* $f^*$，而復雜的model在不同的training data下的分佈比較雜亂(如果data足夠多，那復雜的model也可以得到比較穩定的分佈)

但是如果model比較複雜，那麼每次在不同data下的實驗所得到的不同的 $f^*$ 之間的variance是比較大的，它的散佈就會比較開，就如同下圖中含有高次項的model，每一條 **$f^*$都長得不太像，並且散佈得很開

**那為什麼比較複雜的model，它的散佈就比較開呢？比較簡單的model，它的散佈就比較密集呢？**

原因其實很簡單，其實前面在講regularization正規化的時候也提到了部分原因。簡單的model實際上就是沒有高次項的model，或者高次項的係數非常小的model，這樣的model表現得相當平滑，受到不同的data的影響是比較小的

舉一個很極端的例子，我們的整個model(function set)裡面，就一個function：f=c，這個function只有一個常數項，因此無論training data怎麼變化，從這個最簡單的model裡找出來的 $f^*$ 都是一樣的，它的variance就是等於0

## $f^*$的bias只取決於model的複雜程度

bias是說，我們把所有的 $f^*$ 平均起來得到 **$E(f^*)=\overline{f^*}$*，這個*  $\overline{f^*}$與真值 $\widehat{ f}$ 有多接近

當然這裡會有一個問題是說，總體的真值 $\widehat{f}$ 我們根本就沒有辦法知道，因此這裡只是假設一個 $\widehat{f}$

下面的圖示中，**紅色**線條部分代表5000次實驗分別得到的 $f^*$*，***黑色**線條部分代表真實值 **$\widehat{f}$*，***藍色**線條部分代表5000次實驗得到的 $f^*$ 的平均值 $\overline{f}$

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_5.41.06.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_5.41.06.png)

根據上圖我們發現，當model比較簡單的時候，每次實驗得到的 $f^*$之間的variance會比較小，這些*些* $f^*$會穩定在一個範圍內，但是它們的平均值 $\overline{f}$ 距離真實值 $\widehat{f}$ 會有比較大的偏差；而當model比較複雜的時候，每次實驗得到的 $f^*$之間的variance會比較大，實際體現出來就是每次重新實驗得到的 $f^*$ 都會與之前得到的有較大差距，但是這些差距較大的 $f^*$ 的平均值 $\overline{f}$ 卻和真實值 $\widehat{f}$ 比較接近

上圖分別是含有一次項、三次項和五次項的model做了5000次實驗後的結果，你會發現model越複雜，比如含有5次項的model那一幅圖，每一次實驗得到的 $f^*$ 幾乎是雜亂無章，遍布整幅圖的；但是他們的平均值卻和真實值 $\widehat{f}$ 吻合的很好。也就是說，複雜的model，單次實驗的結果是沒有太大參考價值的，但是如果把考慮多次實驗的結果的平均值，也許會對最終的結果有幫助

註：這裡的單次實驗指的是，用一組training data訓練出model的一組有效參數以構成 $f^*$ (每次獨立實驗使用的training data都是不同的)

### 因此：

- 如果是一個比較簡單的model，那它有比較小的variance和比較大的bias。就像下圖中左下角的打靶模型，每次實驗的 $f^*$ 都比較集中，但是他們平均起來距離靶心會有一段距離(比較適合實驗次數少甚至只有單次實驗的情況)
- 如果是一個比較複雜的model，每次實驗找出來的 $f^*$ 都不一樣，它有比較大的variance但是卻有比較小的bias。就像下圖中右下角的打靶模型，每次實驗的 $f^*$ 都比較分散，但是他們平均起來的位置與靶心比較接近(比較適合多次實驗的情況)

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_5.49.16.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_5.49.16.png)

### 為什麼會這樣？

實際上我們的model就是一個function set，當你定好一個model的時候，實際上就已經定好這個function set的範圍了，那個最好的function只能從這個function set裡面挑出來

如果是一個簡單的model，它的function set的space是比較小的，這個範圍可能根本就沒有包含你的target；如果這個function set沒有包含target，那麼不管怎麼sample，平均起來永遠不可能是target(這裡的space指上圖中左下角那個被model圈起來的空間)

如果這個model比較複雜，那麼這個model所代表的function set的space是比較大的(簡單的model實際上就是複雜model的子集)，那它就很有可能包含target，只是它沒有辦法找到那個target在哪，因為你給的training data不夠，你給的training data每一次都不一樣，所以他每一次找出來的$f^*$都不一樣，但是如果他們是散佈在這個target附近的，那平均起來，實際上就可以得到和target比較接近的位置(這裡的space指上圖中右下角那個被model圈起來的空間)

## Bias vs Variance

由前面的討論可知，比較簡單的model，variance比較小，bias比較大；而比較複雜的model，bias比較小，variance比較大

## bias和variance對error的影響

因此下圖中(也就是之前我們得到的從最高項為一次項到五次項的五個model的error表現)，綠色的線代表variance造成的error，紅色的線代表bias造成的error，藍色的線代表這個model實際觀測到的error

$error_{實際}=error_{variance}+error_{bias}$——藍線為紅線和綠線之和

可以發現，隨著model的逐漸復雜：

- bias逐漸減小，bias所造成的error也逐漸下降，也就是打靶的時候瞄得越來越準，體現為圖中的紅線
- variance逐漸變大，variance所造成的error也逐漸增大，也就是雖然瞄得越來越準，但是每次射出去以後，你的誤差是越來越大的，體現為圖中的綠線
- 當bias和variance這兩項同時被考慮的時候，得到的就是圖中的藍線，也就是實際體現出來的error的變化；實際觀測到的error先是減小然後又增大，因此實際error為最小值的那個點，即為bias和variance的error之和最小的點，就是表現最好的model
- **如果實際error主要來自於variance很大，這個狀況就是overfitting過擬合；如果實際error主要來自於bias很大，這個狀況就是underfitting欠擬合**(可以理解為，overfitting就是過分地包圍了靶心所在的space，而underfitting則是還未曾包圍到靶心所在的space)

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_5.55.03.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_5.55.03.png)

這就是為什麼我們之前要先計算出每一個model對應的error(每一個model都有唯一對應的 $f^*$，因此也有唯一對應的error)，再挑選error最小的model的原因，只有這樣才能綜合考慮bias和variance的影響，找到一個實際error最小的model

### 必須要知道自己的error主要來自於哪裡

### 你現在的問題是bias大，還是variance大？

當你自己在做research的時候，你必須要搞清楚，手頭上的這個model，它目前主要的error是來源於哪裡；你覺得你現在的問題是bias大，還是variance大

你應該先知道這件事情，你才能知道你的future work，你要improve你的model的時候，你應該要走哪一個方向

### 那怎麼知道現在是bias大還是variance大呢？

- 如果model沒有辦法fit training data的examples，代表bias比較大，這時是underfitting

具體來說，就是該model找到的 $f^*$上面並沒有training data的大部分樣本點，如下圖中的linear model，我們只是example抽樣了這幾個藍色的樣本點，而這個model甚至沒有fit這少數幾個藍色的樣本點(這幾個樣本點沒有在 $f^*$ 上)，代表說這個model跟正確的model是有一段差距的，所以這個時候是bias大的情況，是underfitting

- 如果model可以fit training data，在training data上得到小的error，但是在testing data上，卻得到一個大的error，代表variance比較大，這時是overfitting

### 如何針對性地處理bias大 or variance大的情況呢？

遇到bias大或variance大的時候，你其實是要用不同的方式來處理它們

1、**如果bias比較大**

bias大代表，你現在這個model裡面可能根本沒有包含你的target， $\widehat{f}$ 可能根本就不在你的function set裡

對於error主要來自於bias的情況，是由於該model(function set)本來就不好，collect更多的data是沒有用的，必須要從model本身出發

- redesign，重新設計你的model
    - 增加更多的features作為model的input輸入變量

        比如pokemon的例子裡，只考慮進化前cp值可能不夠，還要考慮hp值、species種類...作為model新的input變量

    - 讓model變得更複雜，增加高次項

        比如原本只是linear model，現在考慮增加二次項、三次項...

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_6.00.30.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_6.00.30.png)

2、**如果variance比較大**

- 增加data
    - 如果是5次式，找100個 $f^*$，每次實驗我們只用10只寶可夢的數據訓練model，那我們找出來的100個 $f^*$的散佈就會像下圖一樣雜亂無章；但如果每次實驗我們用100只寶可夢的數據訓練model，那我們找出來的100個 **$f^*$ 的分佈就會像下圖所示一樣，非常地集中；但如果每次實驗我們用100只寶可夢的數據訓練model，那我們找出來的100個 $f^*$的分佈就會像下圖所示一樣，非常地集中
    - 增加data是一個很有效控制variance的方法，假設你variance太大的話，collect data幾乎是一個萬靈丹一樣的東西，並且它不會傷害你的bias
    - 但是它存在一個很大的問題是，實際上並沒有辦法去collect更多的data
    - 如果沒有辦法collect更多的data，其實有一招，根據你對這個問題的理解，自己去generate更多“假的”data
        - 比如手寫數字識別，因為每個人手寫數字的角度都不一樣，那就把所有training data裡面的數字都左轉15°，右轉15°
        - 比如做火車的影像辨識，只有從左邊開過來的火車影像資料，沒有從右邊開過來的火車影像資料，該怎麼辦？實際上可以把每張圖片都左右顛倒，就generate出右邊的火車數據了，這樣就多了一倍data出來
        - 比如做語音辨識的時候，只有男生說的“你好”，沒有女生說的“你好”，那就用男生的聲音用一個變聲器把它轉化一下，這樣男女生的聲音就可以互相轉化，這樣data就可以多出來
        - 比如現在你只有錄音室裡錄下的聲音，但是detection實際要在真實場景下使用的，那你就去真實場景下錄一些噪音加到原本的聲音裡，就可以generate出符合條件的data了
- Regularization(正規化)
    - 就是在loss function裡面再加一個與model高次項係數相關的term，它會希望你的model裡高次項的參數越小越好，也就是說希望你今天找出來的曲線越平滑越好；這個新加的term前面可以有一個weight，代表你希望你的曲線有多平滑
    - 下圖中Regularization部分，左邊第一幅圖是沒有加regularization的test；第二幅圖是加了regularization後的情況，一些怪怪的、很不平滑的曲線就不會再出現，所有曲線都集中在比較平滑的區域；第三幅圖是增加weight的情況，讓曲線變得更平滑
    - 加了regularization以後，因為你強迫所有的曲線都要比較平滑，所以這個時候也會讓你的variance變小；但regularization是可能會傷害bias的，因為它實際上調整了function set的space範圍，變成它只包含那些比較平滑的曲線，這個縮小的space可能沒有包含原先在更大space內的 $\widehat{f}$，因此傷害了bias，所以當你做regularization的時候，需要調整regularization的weight，在variance和bias之間取得平衡

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_6.05.33.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_6.05.33.png)

我們現在會遇到的問題往往是這樣：我們有很多個model可以選擇，還有很多參數可以調，譬如regularization的weight，那通常我們是在bias和variance之間做一些trade-off權衡

我們希望找一個model，它variance夠小，bias也夠小，這兩個合起來給我們最小的testing data的error

### 但是以下這些事情，是你不應該做的：

你手上有training set，有testing set，接下來你想知道model1、model2、model3裡面，應該選哪一個model，然後你就分別用這三個model去訓練出 $f_1^*,f_2^*, f_3^*$，然後把它apply到testing set上面，分別得到三個error為0.9，0.7，0.5，這裡很直覺地會認為是model3最好

但是現在可能的問題是，這個testing set是你自己手上的testing set，是你自己拿來衡量model好壞的testing set，真正的testing set是你沒有的；注意到你自己手上的這筆testing set，它有自己的一個bias(這裡的bias跟之前提到的略有不同，可以理解為自己的testing data跟實際的testing data會有一定的偏差存在)

所以你今天那這個testing set來選擇最好的model的時候，它在真正的testing set上不見得是最好的model，通常是比較差的，所以你實際得到的error是會大於你在自己的testing set上估測到的0.5

以PM2.5預測為例，提供的數據分為training set，public testing set和private testing set三部分，其中public的testing set是供你測試自己的model的，private的testing data是你暫且未知的真正測試數據，現在你的model3在public testing set上的error為0.5，已經成功beat baseline，但是在private的testing set上，你的model3也許根本就沒有beat the baseline，反而是model1和model2可能會表現地更好

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_6.13.58.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_6.13.58.png)

### 怎樣做才是可靠的呢？

### training data分成training set和validation set

你要做的事情是，把你的training set分成兩組：

- 一組是真正拿來training model的，叫做training set(訓練集)
- 另外一組不拿它來training model，而是拿它來選model，叫做validation set(驗證集)

先在training set上找出每個model最好的function $f^*$，然後用validation set來選擇你的model==

也就是說，你手頭上有3個model，你先把這3個model用training set訓練出三個 $f^*$，接下來看一下它們在validation set上的performance

假設現在model3的performance最好，那你可以直接把這個model3的結果拿來apply在testing data上

如果你擔心現在把training set分成training和validation兩部分，感覺training data變少的話，可以這樣做：已經從validation決定model3是最好的model，那就定住model3不變(function的表達式不變) ，然後用全部的data在model3上面再訓練一次(使用全部的data去更新model3表達式的參數)

這個時候，如果你把這個訓練好的model的 $f^*$apply到public testing set上面，你可能會得到一個大於0.5的error，雖然這麼做，你得到的error表面上看起來是比較大的，但是**這個時候你在public set上的error才能夠真正反映你在private set上的error**

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_6.16.28.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_6.16.28.png)

### 考慮真實的測試集

實際上是這樣一個關係：

> training data(訓練集) -> 自己的testing data(測試集) -> 實際的testing data
(該流程沒有考慮自己的testing data的bias)
training set(部分訓練集) -> validation set(部分驗證集) -> 自己的testing data(測試集) -> 實際的testing data
(該流程使用自己的testing data和validation來模擬testing data的bias誤差，可以真實地反映出在實際的data上出現的error)

### 真正的error

當你得到public set上的error的時候(儘管它可能會很大)，不建議回過頭去重新調整model的參數，因為當你再回去重新調整什麼東西的時候，你就又會把public testing set的bias給考慮進去了，這就又回到了第一種關係，即圍繞著有偏差的testing data做model的優化

這樣的話此時你在public set上看到的performance就沒有辦法反映實際在private set上的performance了，因為你的model是針對public set做過優化的，雖然public set上的error數據看起來可能會更好看，但是針對實際未知的private set，這個“優化”帶來的可能是反作用，反而會使實際的error變大

當然，你也許幾乎沒有辦法忍住不去做這件事情，在發paper的時候，有時候你會propose一個方法，那你要attach在benchmark的corpus，如果你在testing set上得到一個差的結果，你也幾乎沒有辦法把持自己不回頭去調一下你的model，你肯定不會只是寫一個paper說這個方法不work這樣子

因此這裡只是說，你要keep in mind，如果在那個benchmark corpus上面所看到的testing的performance，它的error，肯定是大於它在real的application上應該有的值

譬如說你現在常常會聽到說，在image lab的那個corpus上面，error rate都降到3%，那個是超越人類了，但是真的是這樣子嗎？已經有這麼多人玩過這個corpus，已經有這麼多人告訴你說前面這些方法都不work，他們都幫你挑過model了，你已經用“testing” data調過參數了，所以如果你把那些model真的apply到現實生活中，它的error rate肯定是大於3%的

## 如何劃分training set和validation set？

那如果training set和validation set分壞了怎麼辦？如果validation也有怪怪的bias，豈不是對結果很不利？那你要做下面這件事情：

**N-flod Cross Validation**

如果你不相信某一次分train和validation的結果的話，那你就分很多種不同的樣子

譬如說，如果你做3-flod的validation，意思就是你把training set分成三份，你每一次拿其中一份當做validation set，另外兩份當training；分別在每個情境下都計算一下3個model的error，然後計算一下它的average error；然後你會發現在這三個情境下的average error，是model1最好

然後接下來，你就把用整個完整的training data重新訓練一遍model1的參數；然後再去testing data上test

原則上是，如果你少去根據public testing set上的error調整model的話，那你在private testing set上面得到的error往往是比較接近public testing set上的error

![%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_6.20.32.png](%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%20%EF%BC%ADachine%20Learning%20f4e96fe86d7f45ebba99771d5eb1ed4b/_2020-09-06_6.20.32.png)

1、一般來說，error是bias和variance共同作用的結果

2、model比較簡單和比較複雜的情況：

- 當model比較簡單的時候，variance比較小，bias比較大，此時 $f^*$會比較集中，但是function set可能並沒有包含真實值 $\widehat{f}$；此時model受bias影響較大
- 當model比較複雜的時候，bias比較小，variance比較大，此時function set會包含真實值 $\widehat{f}$，但是 $f^*$ 會比較分散；此時model受variance影響較大

3、區分bias大 or variance大的情況

- 如果連採樣的樣本點都沒有大部分在model訓練出來的 $f^*$ 上，說明這個model太簡單，bias比較大，是欠擬合
- 如果樣本點基本都在model訓練出來的 $f^*$ 上，但是testing data上測試得到的error很大，說明這個model太複雜，variance比較大，是過擬合

4、bias大 or variance大的情況下該如何處理

- 當bias比較大時，需要做的是重新設計model，包括考慮添加新的input變量，考慮幫model添加高次項；然後對每一個model對應的 $f^*$ 計算出error，選擇error值最小的model(隨model變複雜，bias會減小，variance會增加，因此這裡分別計算error，取兩者平衡點)
- 當variance比較大時，一個很好的辦法是增加data(可以憑藉經驗自己generate data)，當data數量足夠時，得到的 $f^*$ 實際上是比較集中的；如果現實中沒有辦法collect更多的data，那麼就採用regularization正規化的方法，以曲線的平滑度為條件控制function set的範圍，用weight控制平滑度閾值，使得最終的model既包含 $\widehat{f}$，variance又不會太大

5、如何選擇model

- 選擇model的時候呢，我們手頭上的testing data與真實的testing data之間是存在偏差的，因此我們要將training data分成training set和validation set兩部分，經過validation挑選出來的model再用全部的training data訓練一遍參數，最後用testing data去測試error，這樣得到的error是模擬過testing bias的error，與實際情況下的error會比較符合