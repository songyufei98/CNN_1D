import os
import time
import pickle
import torch
import read_data
import numpy as np
from torch.autograd import Variable
from model.CNN1D import CNN_1D
from tqdm import tqdm
from utils import plot_save_lsm
from torch.utils.data import DataLoader, TensorDataset
import config

config = config.config


def save_LSM():
    plot_and_save = True
    print('*******************************************generate LSM*******************************************')
    model = CNN_1D().to(config["device"])
    model.load_state_dict(torch.load(os.path.join('Result', 'CNN_1D_0_1_FR', 'best.pth')))

    tensor_data = read_data.get_feature_data()
    print('size of the study area：' + str(tensor_data.shape))
    creat = read_data.creat_dataset(tensor_data)
    data = creat.creat_new_tensor()
    images_list = []
    probs = []
    model.eval()
    with torch.no_grad():   
        for i in range(config["height"]):
            for j in range(config["width"]):
                # 读取因子向量
                images_list.append(data[:, i, j].astype(np.float32))
                if (i != 0 and i % config["Cutting_window"] == 0 and j == config["width"] - 1) or ( 
                    i == config["height"] - 1 and j == config["width"] - 1):
                    start_time = time.time()
                    print('i=' + str(i) + ' j=' + str(j))
                    pred_data = np.stack(images_list)
                    images_list = []
                    pred_dataset = TensorDataset(torch.from_numpy(pred_data))
                    pred_loader = DataLoader(dataset=pred_dataset, batch_size=config["batch_size"], shuffle=False)
                    for images in tqdm(pred_loader):
                        images = torch.stack([image.unsqueeze(1).cuda() for image in images])
                        images = Variable(images.squeeze(0)).to(config["device"])
                        probs.append(torch.softmax(model(images).cpu(), dim=1)[:, 1])
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    minutes = int(elapsed_time // 60)
                    seconds = elapsed_time % 60
                    print(f"滑动窗口处理时间: {minutes} 分钟 {seconds} 秒")
    probs = np.concatenate(probs)
    print('概率列表生成完成')
    if plot_and_save:
        plot_save_lsm(os.path.join('Result', 'CNN_1D_0_1_FR'), probs)
    read_data.save_to_tif(probs, os.path.join('Result', 'CNN_1D_0_1_FR', 'CNN1D_LSM.tif'))
            