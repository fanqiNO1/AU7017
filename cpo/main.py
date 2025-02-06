import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from mmengine.hooks import CheckpointHook
from mmengine.logging import print_log
from mmengine.model import BaseModel
from mmengine.optim import CosineAnnealingLR, LinearLR
from mmengine.runner import Runner
from model import VisionTransformer
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class Model(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = VisionTransformer(image_size=28,
                                       num_layers=6,
                                       num_classes=10,
                                       patch_size=7,
                                       in_channels=1,
                                       hidden_size=64,
                                       num_attention_heads=8,
                                       num_key_value_heads=2,
                                       intermediate_size=128,
                                       act_fn=nn.SiLU())

    def loss(self, output, y):
        logits = torch.log_softmax(output, dim=-1)
        # get y_reject
        with torch.no_grad():
            y_chosen = y.unsqueeze(-1)
            reject_logits = torch.scatter(logits,
                                          dim=-1,
                                          index=y_chosen,
                                          value=-torch.inf)
            y_reject = torch.argmax(reject_logits, dim=-1, keepdim=True)
        # compute loss
        probs_chosen = torch.gather(logits, dim=-1, index=y_chosen).squeeze(-1)
        probs_reject = torch.gather(logits, dim=-1, index=y_reject).squeeze(-1)
        loss = -F.logsigmoid(probs_chosen - probs_reject).mean()
        return loss

    def log_params(self):
        params = sum(p.numel() for p in self.model.parameters())
        if params > 1e6:
            params = f'{params / 1e6:.2f}M'
        elif params > 1e3:
            params = f'{params / 1e3:.2f}K'
        print_log(f'Number of parameters: {params}', logger='current')

    def forward(self, x, y, mode):
        output = self.model(x)
        if mode == 'loss':
            return {'loss': self.loss(output, y)}
        elif mode == 'predict':
            return output, y


class Accuracy(BaseMetric):

    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        return dict(accuracy=100 * total_correct / total_size)


def main():
    model = Model()

    train_set = MNIST(root='./data',
                      train=True,
                      download=True,
                      transform=ToTensor())
    test_set = MNIST(root='./data',
                     train=False,
                     download=True,
                     transform=ToTensor())

    train_loader = dict(batch_size=32,
                        num_workers=0,
                        dataset=train_set,
                        sampler=dict(type='DefaultSampler', shuffle=True),
                        collate_fn=dict(type='default_collate'))
    test_loader = dict(batch_size=32,
                       num_workers=0,
                       dataset=test_set,
                       sampler=dict(type='DefaultSampler', shuffle=False),
                       collate_fn=dict(type='default_collate'))

    runner = Runner(
        model=model,
        work_dir=f'./ckpts/cpo/{str(model.model)}',
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        train_cfg=dict(by_epoch=True, max_epochs=10, val_interval=1),
        val_cfg=dict(),
        optim_wrapper=dict(optimizer=dict(type=torch.optim.AdamW, lr=1e-3)),
        param_scheduler=[
            dict(type=LinearLR,
                 start_factor=1e-3,
                 by_epoch=True,
                 begin=0,
                 end=2,
                 convert_to_iter_based=True),
            dict(type=CosineAnnealingLR,
                 eta_min=0.0,
                 by_epoch=True,
                 begin=2,
                 end=10,
                 convert_to_iter_based=True)
        ],
        val_evaluator=dict(type=Accuracy),
        default_hooks=dict(checkpoint=dict(
            type=CheckpointHook, max_keep_ckpts=1, save_best='auto')),
        randomness=dict(seed=0x66ccff))
    model.log_params()
    runner.train()


if __name__ == '__main__':
    main()
