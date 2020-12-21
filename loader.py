import torchvision.transforms as transforms
import torchvision
import torch

class Loader:
    
    TARGET_DIR  =   "test-dataset/"

    def __init__(self):
        pass 

    def setDir(self,targetDir):
        self.TARGET_DIR =   targetDir
    
    def dataloader(self,imgSize):
        targetData  =   torchvision.datasets.ImageFolder(
            root=self.TARGET_DIR,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((imgSize,imgSize)),
                transforms.ToTensor()
            ])
        )

        return torch.utils.data.DataLoader(
            targetData,
            batch_size=1,
            shuffle=True
        )