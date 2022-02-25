import cv2
from PIL import Image
import argparse
import numpy as np
import pandas as pd
import h5py
import time

import torch
from torch.utils.data import Dataset, DataLoader 
import torchvision


def look_id_mapping(mapping_path):
    look_mapping = pd.read_csv(mapping_path)
    mapping = []
    for i in range(len(look_mapping)):
        mapping.append((
            look_mapping["user_id"][i], 
            look_mapping["look_id"][i],
            look_mapping["idx"][i]
        ))
    return mapping

class ImageDataset(Dataset):
    def __init__(self, hdf5_path, mapping_path):
        super(Dataset, self).__init__()
        self.transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize([1024, 1024]),
            torchvision.transforms.ToTensor()
            ])
        self.data = h5py.File(hdf5_path, 'r')
        self.mapping = look_id_mapping(mapping_path)

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index):
        user_id, look_id, idx = self.mapping[index]
        group = self.data[str(user_id)][:]
        img = np.array(group[idx,:,:,:])
        img = Image.fromarray(img, 'RGB')
        img = self.transform(img)
        return user_id, look_id, idx, img


class HumanDetection:

    def __init__(self, opt):
        self.opt = opt
        dataset = ImageDataset(self.opt.hdf5_path, self.opt.mapping_path)
        self.n_images = len(dataset)
        print("There are %d images" % self.n_images)

        self.data_loader = DataLoader(
            dataset, 
            batch_size=self.opt.batch_size, 
            num_workers=self.opt.num_workers, shuffle=False, drop_last=False
        )

    def run(self):
        """
        run the human detection model
        """
        start = time.time()

        print("CUDA available: %s" % torch.cuda.is_available())
        if self.opt.use_cuda == -1:
            print("use cpu")
            device = torch.device("cpu")
        else:
            print("use cuda: %d" % self.opt.use_cuda)
            device = torch.device("cuda:%d" % (self.opt.use_cuda))
        
        print("load faster r-cnn model")
        model = self.load_model(device)

        print("start cropping human regions by batch")
        batch_no = 0
        
        for user_ids, look_ids, _ , imgs in self.data_loader:
            if batch_no % 50 == 0:
                print("Batch %d / %d" % (batch_no, int(self.n_images / self.opt.batch_size)))

            self.crop(device, model, user_ids, look_ids, imgs)
            batch_no += 1

        end = time.time()
        print("Total time used: %.2f sec" % (end-start))

    def load_model(self, device, pretrained=True, model_save_path=None):
        """
        load pretrain Faster R-CNN model
        @param device: cuda:0 or cpu
        @param pretrained: whether the model is pretrained or not
        @param model_save_path: path where the model is saved
        return: downloaded model
        """
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=False)
        if model_save_path:
            model.load_state_dict(torch.load(model_save_path))
        model = model.to(device)
        model.eval()
        return model

    def tensor_to_image(self, arr):
        """
        helper function to convert tensor to cv2 image
        """
        arr = arr * 255
        arr = arr.astype('float32')
        arr = np.transpose(arr, (1, 2, 0))
        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) 
        return img

    def crop(self, device, model, user_ids, look_ids, imgs, conf_thres=0.8):
        """
        crop the largest detected human from the image
        @param device: cuda:0 or cpu
        @param model: pretrain model
        @param user_ids: batch of user id
        @param look_ids: batch of look id
        @param imgs: batch of images
        """
        pred = model(imgs.to(device))
        
        if self.opt.use_cuda != -1:
            boxes = [pred[i]["boxes"].cpu().detach().numpy() for i in range(len(imgs))]
            classes = [pred[i]["labels"].cpu().numpy() for i in range(len(imgs))]
            scores = [pred[i]["scores"].cpu().detach().numpy() for i in range(len(imgs))]
            imgs = imgs.cpu().detach().numpy()
            user_ids = user_ids.cpu().detach().numpy()
            look_ids = look_ids.cpu().detach().numpy()
        else:
            boxes = [pred[i]["boxes"].detach().numpy() for i in range(len(imgs))]
            classes = [pred[i]["labels"].numpy() for i in range(len(imgs))]
            scores = [pred[i]["scores"].detach().numpy() for i in range(len(imgs))]
            imgs = imgs.detach().numpy()
            user_ids = user_ids.detach().numpy()
            look_ids = look_ids.detach().numpy()
        
        # save the img of the largest detected human region
        for i in range(len(imgs)):
            biggest_bounding_box = None
            greatest_area = 0

            for j in range(len(boxes[i])):
                # filter class = 1 (person)
                if classes[i][j] == 1:
                    area = (boxes[i][j][3] - boxes[i][j][1]) * (boxes[i][j][2] - boxes[i][j][0])
                    if area > greatest_area and scores[i][j] > conf_thres:
                        greatest_area = area
                        biggest_bounding_box = boxes[i][j]

            if biggest_bounding_box is not None:
                x1, y1, x2, y2 = list(map(int, biggest_bounding_box))
                single_person_image = imgs[i][:, y1:y2, x1:x2]
                single_person_image = self.tensor_to_image(single_person_image)
                img_path = "%s/%d_%d.jpg" % (self.opt.save_path, user_ids[i], look_ids[i])
                cv2.imwrite(img_path, single_person_image)
                print("write to %s" % img_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", type=int, default=0, help="-1 not use cuda, otherwise is the cuda idx, from 0 to 3")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers")
    parser.add_argument("--mapping_path", type=str, help="Path to load mapping csv")
    parser.add_argument("--hdf5_path", type=str, help="Path to load hdf5 file")
    parser.add_argument("--save_path", type=str, help="Path to save the detected human images")
    opt = parser.parse_args()

    model = HumanDetection(opt)
    model.run()
