import os
import re
import json
import torch
import torch.nn as nn
from tqdm import tqdm  # 导入 tqdm 进度条模块
import numpy as np
from dataset.wwadl_test import WWADLDatasetTestSingle
from strategy.evaluation.softnms import softnms_v2
from strategy.evaluation.eval_detection import ANETdetection
# from dataset.action import id_to_action

class Tester(object):
    def __init__(self,
                 config,
                 test_dataset,
                 model,
                 pt_file_name = None
                 ):
        super(Tester, self).__init__()
        self.model = model
        self.test_dataset = test_dataset
        self.checkpoint_path = config['path']['result_path']

        self.clip_length = config['dataset']['clip_length']
        self.num_classes = config['model']['num_classes']

        self.top_k = config['testing']['top_k']
        self.conf_thresh = config['testing']['conf_thresh']
        self.nms_thresh = config['testing']['nms_thresh']
        self.nms_sigma = config['testing']['nms_sigma']

        self.eval_gt = test_dataset.eval_gt
        # self.eval_gt = '/root/shared-nvme/dataset/all_30_3/imutrain_annotations.json'
        self.id_to_action = test_dataset.id_to_action
        print(self.id_to_action)

        if pt_file_name is None:
            pt_file_name = self.get_latest_checkpoint()

        print(pt_file_name)

        self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, pt_file_name)))  # 加载模型权重

        self.pt_file_name = pt_file_name

    def get_latest_checkpoint(self):
        # 获取目录中的所有文件
        all_files = os.listdir(self.checkpoint_path)
        print(all_files)
        # 正则表达式匹配文件名格式
        pattern = re.compile(r".*-epoch-(\d+)\.pt$")

        # 保存符合条件的文件和其对应的epoch
        valid_files = []
        for file in all_files:
            match = pattern.match(file)
            if match:
                epoch = int(match.group(1))
                valid_files.append((file, epoch))

        print(valid_files)

        # 找到epoch最大的文件
        if valid_files:
            latest_file = max(valid_files, key=lambda x: x[1])
            return latest_file[0]  # 返回文件名
        else:
            return None  # 如果没有符合条件的文件

    def _to_var(self, data):
        for key, value in data.items():
            data[key] = value.unsqueeze(0).cuda()  # Directly move tensor to device
        return data

    def testing(self):

        self.model.eval().cuda()  # 切换到 eval 模式，并将模型移到 GPU 上
        score_func = nn.Softmax(dim=-1)  # 使用 Softmax 将分类得分转换为概率
        result_dict = {}  # 存储最终结果

        test_files = list(self.test_dataset.dataset())  # 确保数据集可以多次遍历
        for file_name, data_iterator in tqdm(test_files, desc="Testing Progress", unit="file"):

            output = [[] for _ in range(self.num_classes)]  # 初始化每个类别的输出结果
            res = torch.zeros(self.num_classes, self.top_k, 3)  # 用于存储 Soft-NMS 处理后的 top-k 结果

            for clip, segment in data_iterator:
                clip = self._to_var(clip)
                # clip = clip.unsqueeze(0).cuda()  # 添加 batch 维度，并移动到 GPU
                with torch.no_grad():  # 禁用梯度计算
                    output_dict = self.model(clip)  # 模型推理

                loc, conf, priors = output_dict['loc'][0], output_dict['conf'][0], output_dict['priors'][0]

                decoded_segments = torch.cat(
                    [priors[:, :1] * self.clip_length - loc[:, :1],  # 左边界
                     priors[:, :1] * self.clip_length + loc[:, 1:]], dim=-1)  # 右边界
                decoded_segments.clamp_(min=0, max=self.clip_length)  # 裁剪到合法范围

                conf = score_func(conf)  # 使用 Softmax 计算分类概率
                conf = conf.view(-1, self.num_classes).transpose(1, 0)  # 转换形状
                conf_scores = conf.clone()  # 复制分类结果

                # 筛选满足置信度阈值的检测结果
                for cl in range(0, self.num_classes):  # 遍历每个类别
                    c_mask = conf_scores[cl] > self.conf_thresh  # 筛选置信度高的结果
                    scores = conf_scores[cl][c_mask]
                    if scores.size(0) == 0:  # 如果没有满足阈值的结果，跳过
                        continue
                    l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
                    segments = decoded_segments[l_mask].view(-1, 2)
                    segments = segments + segment[0]  # 转换为时间区间
                    segments = torch.cat([segments, scores.unsqueeze(1)], -1)  # 拼接区间和分数
                    output[cl].append(segments)  # 保存到输出列表

            # 对每个类别应用 Soft-NMS
            sum_count = 0
            for cl in range(0, self.num_classes):
                if len(output[cl]) == 0:
                    continue
                tmp = torch.cat(output[cl], 0)  # 合并所有片段
                tmp, count = softnms_v2(tmp, sigma=self.nms_sigma, top_k=self.top_k)  # 进行 Soft-NMS
                res[cl, :count] = tmp  # 保存处理后的 top-k 结果
                sum_count += count

            sum_count = min(sum_count, self.top_k)  # 限制最大数量
            flt = res.contiguous().view(-1, 3)
            flt = flt.view(self.num_classes, -1, 3)  # 重新组织结果

            # 生成 JSON 格式的结果
            proposal_list = []
            for cl in range(0, self.num_classes):  # 遍历每个类别
                class_name = self.id_to_action[str(cl)]  # 获取类别名称
                tmp = flt[cl].contiguous()
                tmp = tmp[(tmp[:, 2] > 0).unsqueeze(-1).expand_as(tmp)].view(-1, 3)  # 筛选有效结果
                if tmp.size(0) == 0:
                    continue
                tmp = tmp.detach().cpu().numpy()
                for i in range(tmp.shape[0]):
                    tmp_proposal = {
                        'label': class_name,
                        'score': float(tmp[i, 2]),
                        'segment': [float(tmp[i, 0]), float(tmp[i, 1])]
                    }
                    proposal_list.append(tmp_proposal)

            result_dict[file_name] = proposal_list  # 保存视频结果

        # 保存最终结果为 JSON 文件
        output_dict = {"version": "THUMOS14", "results": dict(result_dict), "external_data": {}}
        json_name = "checkpoint_" + str(self.pt_file_name) + ".json"
        with open(os.path.join(self.checkpoint_path, json_name), "w") as out:
            json.dump(output_dict, out)
        self.eval_pr = os.path.join(self.checkpoint_path, json_name)

        self.eval()

    def eval(self):
        """
        Evaluate model performance and save a report to self.checkpoint_path directory.
        """
        # Define tIoU thresholds
        tious = np.linspace(0.5, 0.95, 10)

        # Initialize ANETdetection
        anet_detection = ANETdetection(
            ground_truth_filename=self.eval_gt,
            prediction_filename=self.eval_pr,
            subset='test',
            tiou_thresholds=tious
        )

        # Perform evaluation
        mAPs, average_mAP, ap = anet_detection.evaluate()

        

        # Prepare report content
        report_lines = []
        report_lines.append("Evaluation Report")
        report_lines.append("===================")
        report_lines.append(f"Evaluation Ground Truth: {self.eval_gt}")
        report_lines.append(f"Evaluation Predictions: {self.eval_pr}")
        report_lines.append("\nResults:")
        for (tiou, mAP) in zip(tious, mAPs):
            report_lines.append(f"mAP at tIoU {tiou:.1f}: {mAP:.4f}")
        report_lines.append(f"\nAverage mAP: {average_mAP:.4f}")
        # Convert report content to string
        report_content = "\n".join(report_lines)

        # Define report file path
        report_filename = os.path.join(self.checkpoint_path, "evaluation_report.txt")

        # Save report to file
        try:
            with open(report_filename, "w") as report_file:
                report_file.write(report_content)
            print(f"Evaluation report saved to: {report_filename}")
        except Exception as e:
            print(f"Error saving evaluation report: {e}")


if __name__ == '__main__':

    from global_config import config
    from model import wifiTAD, wifiTAD_config

    model_config = wifiTAD_config(config['model']['model_set'])
    model = wifiTAD(model_config)

    dataset = WWADLDatasetTestSingle(dataset_dir='/root/shared-nvme/dataset/imu_30_3')
    test = Tester(config,dataset, model)
    test.testing()