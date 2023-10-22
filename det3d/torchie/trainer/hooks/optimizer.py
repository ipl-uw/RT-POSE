from torch.nn.utils import clip_grad
from .hook import Hook
import torch

class OptimizerHook(Hook):
    def __init__(self, grad_clip=None, enable_amp=False):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip
        )

    def after_train_iter(self, trainer):
        if not torch.isfinite(trainer.outputs['loss']):
            trainer.logger.CRITICAL('The loss DIVERGED')
            return
        trainer.optimizer.zero_grad()
        trainer.scaler.scale(trainer.outputs["loss"]).backward()
        if self.grad_clip is not None:
            trainer.scaler.unscale_(trainer.optimizer)
            self.clip_grads(trainer.model.parameters())
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()