from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy 
from .. import builder
from det3d.models.losses.jde_loss import JDELoss
from det3d.models.bbox_heads.center_head import SepHead
from torch import nn

@DETECTORS.register_module
class RadarNetSingleStage(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        jde_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(RadarNetSingleStage, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        if jde_cfg.pop('enable', False):
            self.jde_loss = JDELoss(**jde_cfg)
            self.jde_weight = jde_cfg.weight
            emb_head_cfg = jde_cfg.emb_head_cfg
            self.num_classes = [len(t["class_names"]) for t in emb_head_cfg.tasks]
            self.class_names = [t["class_names"] for t in emb_head_cfg.tasks]
            self.tasks = nn.ModuleList()
            for _ in self.num_classes:
                self.tasks.append(SepHead(emb_head_cfg.share_conv_channel, emb_head_cfg.head, bn=True, final_kernel=3))
        else:
            self.jde_loss = None


    def extract_feat(self, data):
        input_features = self.reader(data['rdr_tensor'])
        x = self.backbone(
                input_features
            )
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        x = self.extract_feat(example)
        preds, shared_conv_feat = self.bbox_head(x)
        app_emb_tasks = None
        if self.jde_loss is not None:
            app_emb_tasks = []
            for task_id, task in enumerate(self.tasks):
                app_emb_tasks.append(task(shared_conv_feat)['emb'])
        if return_loss:
            loss_dict = self.bbox_head.loss(example, preds, self.test_cfg)
            if self.jde_loss is not None:
                jde_loss, total_pos, total_neg = [], 0, 0
                for task_id, task in enumerate(self.tasks):
                    loss, num_pos_task, num_neg_task = self.jde_loss(app_emb_tasks[task_id],\
                    example['mask'][task_id], example['ind'][task_id], example['obj_id'][task_id])
                    total_pos += num_pos_task
                    total_neg += num_neg_task
                    jde_loss.append(loss)
                # sum jde loss from all tasks and all batches
                jde_loss = [loss for jde_loss_task in jde_loss for loss in jde_loss_task]
                jde_loss = sum(jde_loss)
                jde_loss *= self.jde_weight
                loss_dict.update({'jde_loss': jde_loss, 'num_pos': total_pos, 'num_neg': total_neg})
            return loss_dict
        
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg, app_emb_tasks)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        # todo: implement this
        x, voxel_feature = self.extract_feat(example)
        bev_feature = x 
        preds, final_feat = self.bbox_head(x)

        if return_loss:
            # manual deepcopy ...
            new_preds = []
            for pred in preds:
                new_pred = {} 
                for k, v in pred.items():
                    new_pred[k] = v.detach()
                new_preds.append(new_pred)

            boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

            return boxes, bev_feature, voxel_feature, final_feat, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            boxes = self.bbox_head.predict(example, preds, self.test_cfg)
            return boxes, bev_feature, voxel_feature, final_feat, None 
        