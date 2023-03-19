import PIL.ImageOps
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 37)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def prepare_image(img: Image) -> Image:
    return img \
        .transpose(Image.FLIP_LEFT_RIGHT) \
        .transpose(Image.ROTATE_90) \
        .resize((28, 28), Image.ANTIALIAS)


file_name = 'test.png'
img = Image.open(file_name)
img = img.convert('L')
img = prepare_image(img)
img = PIL.ImageOps.invert(img)

plt.imshow(img)
plt.show()

train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

img = train_transform(img)

img = torch.unsqueeze(img, dim=0)

model = CNN()
model.load_state_dict(torch.load('./model/CNN_letter.pk', map_location='cpu'))
model.eval()

index_to_class = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'
    , 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

with torch.no_grad():
    y = model(img)
    output = torch.squeeze(y)

    predict = torch.softmax(output, dim=0)

    predict_cla = torch.argmax(predict).numpy()
    print("类别：", predict_cla)
print("预测字母为", index_to_class[predict_cla], "置信度为", predict[predict_cla].numpy())
