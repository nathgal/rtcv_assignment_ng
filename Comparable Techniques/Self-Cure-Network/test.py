import torchvision.models as models
from torchvision import transforms
import cv2

from torch.autograd import Variable
import torch
import torch.nn as nn

class Res18Feature(nn.Module):
    def __init__(self, pretrained, num_classes = 7):
        super(Res18Feature, self).__init__()
        resnet = models.resnet18(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2])  # before avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # After avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512

        self.fc = nn.Linear(fc_in_dim, num_classes)  # New fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        attention_weights = self.alpha(x)
        out = attention_weights * self.fc(x)
        return attention_weights, out

# Variables

model_save_path = "C:/Users/natha/Desktop/UM/Research Topics in Computer Vision/Assignment/Implementation/SOTA/ijba_res18_naive.pth.tar"
img_path = "C:/Users/natha/Desktop/RAF-DB/basic/Image/original/test_0003.jpg"

# ------------------------ Download Data --------------------------- #

preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        
res18 = Res18Feature(pretrained = False)
checkpoint = torch.load(model_save_path)
res18.load_state_dict(checkpoint['state_dict'], strict=False)
res18.cuda()
res18.eval()

for i in range(6):

    image = cv2.imread(img_path)
    image = image[:, :, ::-1]  # BGR to RGB
    image_tensor = preprocess_transform(image)
    print(image_tensor.shape)
    tensor = Variable(torch.unsqueeze(image_tensor, dim=0).float(), requires_grad=False)

    print(tensor.shape)  # [1,3, 224, 224]
    tensor=tensor.cuda()
    print(tensor.shape)

    _, outputs = res18(tensor)
    _, predicts = torch.max(outputs, 1)

    print(outputs)
    print(predicts)

