# LeNet-5-EMNIST

基于LeNet-5卷积神经网络与EMNIST数据集实现手写英文字母识别。

# 实现方法

仿照MNIST的实现方法，下载EMNIST，调参数。其实主要就是卷积核大小、卷积核个数(特征图需要多少个)、池化核大小(采样率多少)，这些数据弄懂了就可以顺利调制参数。

# 注意事项

EMNIST数据集的标准图片和MNIST一样是黑底白字，但是EMNIST的标准图片要水平翻转，然后顺时针旋转90度，否则识别都是错误的。

```
def prepare_image(img: Image) -> Image:
    return img \
        .transpose(Image.FLIP_LEFT_RIGHT) \
        .transpose(Image.ROTATE_90) \
        .resize((28, 28), Image.ANTIALIAS)
```

# 使用方法

1、拉取项目

2、如果想重新训练数据，修改 EMNIST.py 中

```
EPOCH = 5 #训练整批数据次数，训练次数越多，精度越高
BATCH_SIZE = 50 #每次训练的数据集个数
```

然后运行 EMNIST.py 。

3、 使用画图工具修改 test.png 。比如想测试手写字母a，就在 test.png 画个a。

4、运行 EMNIST_test.py ，控制台得出预测结果
