from torch.utils.data import Dataset
from torchvision import transforms
from numpy import genfromtxt
from os import path, listdir
from PIL import Image
import torch
from matplotlib import pyplot as plt

img_resize = 256
img_crop = 224
filenameToPILImage = lambda x: Image.open(x)

class BenedettiniDataset(Dataset):
    def __init__(self, root_dir, support_set_size=35):
        self.root_dir = root_dir
        self.support_dir = path.join(self.root_dir, 'Training')
        self.target_dir = path.join(self.root_dir, 'Test')
        self.support_set_size = support_set_size
        self.support_dict = self.createSupportDict()
        self.target_len, self.target_list = self.createTargetList()

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize(img_resize),
            transforms.CenterCrop(img_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def createTargetList(self):
        target_list = []
        for i in range(100, 167):
            try:
                images_dir = path.join(self.target_dir, str(i))
                target_labels = genfromtxt(path.join(self.target_dir, 'labels/%d.txt' % i))
                target_images = [filename for filename in listdir(images_dir)]
        
                if len(target_labels) != len(target_images):
                    print("error labels and images num are not the same!")
                    raise Exception

                for index, target_path in enumerate(target_images):
                    target_img_path = path.join(str(i), target_path)
                    target_class = int(target_labels[index])
                    target_list.append((target_img_path, target_class))

            except Exception as e:
                pass

        return len(target_list), target_list

    def createSupportDict(self):
        labels = genfromtxt(path.join(self.root_dir, 'labels.txt'), delimiter='\t')
        labels = labels[1:] # drop field names
        labels_dict = {int(a) : b for b,a in labels}
        
        support_dict = {}
        for key, value in labels_dict.items():
            support_dict[key] = [filename for filename in listdir(self.support_dir) if filename.startswith(str(value))][0]
        
        return support_dict
        
    def getSupportSet(self):
        support = torch.FloatTensor(self.support_set_size, 3, img_crop, img_crop)
        #create support
        for label, support_image_path in self.support_dict.items():
            support_image = self.transform(path.join(self.support_dir, support_image_path))
            support[label] = support_image
        return support

    def __getitem__(self, index):
        #create target
        target_image_path, gt = self.target_list[index]
        target_image = self.transform(path.join(self.target_dir, target_image_path))
        return target_image, gt

    def __len__(self):
        return self.target_len

import math
def show_item(target, support_set, gt, n_rows):
    def cround(x):
        res = x - math.floor(x)
        return math.ceil(x) if res != 0 else math.floor(x)
    
    pilTrans = transforms.ToPILImage()
    fig = plt.figure(figsize=(10, 10))#constrained_layout=True)
    n_cols = cround(support_set.shape[0]/n_rows)
    gs = fig.add_gridspec(n_rows+1, n_cols)
    
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(pilTrans(target))
    ax.axis('off')
    ax.title.set_text('GT = '+str(gt))
        
    r = 1
    c = 0
    for idx, sup_im in enumerate(support_set):
        ax = fig.add_subplot(gs[r, c])
        c+=1
        if c == n_cols:
            c=0
            r+=1
        ax.imshow(pilTrans(sup_im))
        ax.axis('off')
        ax.title.set_text(str(idx))
    plt.show()

if __name__ == "__main__":
    b = BenedettiniDataset(r"E:\Sources\Repos\filtered-benedettini\Object Retrieval\Monastero dei Benedettini")
    support = b.getSupportSet()
    for i in range(len(b)):
        target, gt = b[i]
        show_item(target, support, gt, 5)