import multiprocessing
import os
import random
from typing import Any, List, Optional, Tuple, Union
from functools import partial

import aim
import albumentations as A
import fire
import numpy as np
import torch
from aim.pytorch_ignite import AimLogger
from albumentations.pytorch.transforms import ToTensorV2
from mean_ap import CocoMetric, convert_to_coco_api
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import VOCDetection
from torchvision.models import detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.utils import draw_bounding_boxes

import ignite.distributed as idist

from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.distributed.utils import one_rank_only
from ignite.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.metrics.running_average import RunningAverage

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

AVAILABLE_MODELS = [
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "retinanet_resnet50_fpn",
]


class Dataset(VOCDetection):

    classes = (
        "aeroplane",
        "bicycle",
        "boat",
        "bus",
        "car",
        "motorbike",
        "train",
        "bottle",
        "chair",
        "diningtable",
        "pottedplant",
        "sofa",
        "tvmonitor",
        "bird",
        "cat",
        "cow",
        "dog",
        "horse",
        "sheep",
        "person",
    )
    name2class = {v: k + 1 for k, v in enumerate(classes)}
    class2name = {k + 1: v for k, v in enumerate(classes)}

    def __init__(self, transforms: A.BasicTransform, *args, **kwargs):
        self.albu_transform = transforms
        super().__init__(*args, **kwargs)
        self.transforms = partial(self._transforms, self)

    @staticmethod
    def _transforms(self, img, target):
        # make transforms a staticmethod to prevent
        # maximum recursion of `repr` method
        annotation = target['annotation']
        bbox_classes = list(
            map(
                lambda x: (
                    int(x["bndbox"]["xmin"]),
                    int(x["bndbox"]["ymin"]),
                    int(x["bndbox"]["xmax"]),
                    int(x["bndbox"]["ymax"]),
                    x["name"],
                ),
                annotation["object"],
            )
        )
        result = self.albu_transform(image=np.array(img), bboxes=bbox_classes)
        annotation["bboxes"] = result["bboxes"]
        return result['image'] / 255.0, annotation

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, annotation = super().__getitem__(index)
        bboxes = np.stack([a[:4] for a in annotation["bboxes"]])
        labels = [self.name2class[a[4]] for a in annotation["bboxes"]]
        target = {}
        target["boxes"] = torch.tensor(bboxes)
        target["labels"] = torch.tensor(labels)
        target["image_id"] = annotation['filename']
        target["area"] = torch.tensor((bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0]))
        target["iscrowd"] = torch.tensor([False] * len(bboxes))

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def run(
    local_rank: int,
    device: str,
    experiment_name: str,
    dataset_root: str = "./dataset",
    log_dir: str = "./log",
    model: str = "fasterrcnn_resnet50_fpn",
    epochs: int = 13,
    batch_size: int = 4,
    lr: float = 0.01,
    download: bool = False,
    resume_from: Optional[dict] = None,
    visualize_images: int = 16,
) -> None:
    bbox_params = A.BboxParams(format="pascal_voc")
    train_transform = A.Compose(
        [A.HorizontalFlip(p=0.5), ToTensorV2()],
        bbox_params=bbox_params,
    )
    val_transform = A.Compose([ToTensorV2()], bbox_params=bbox_params)

    download = local_rank == 0 and download
    train_dataset = Dataset(root=dataset_root, download=download, image_set="train", transforms=train_transform)
    val_dataset = Dataset(root=dataset_root, download=download, image_set="val", transforms=val_transform)
    vis_dataset = Subset(val_dataset, random.sample(range(len(val_dataset)), k=visualize_images))

    train_dataloader = idist.auto_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    vis_dataloader = DataLoader(vis_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = idist.auto_model(model)
    scaler = GradScaler()
    optimizer = SGD(lr=lr, params=model.parameters())
    optimizer = idist.auto_optim(optimizer)
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=len(train_dataloader) * epochs)

    def update_model(engine, batch):
        model.train()
        images, targets = batch
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]

        with torch.autocast(device, enabled=True):
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_items = {k: v.item() for k, v in loss_dict.items()}
        loss_items["loss_average"] = loss.item() / 4

        return loss_items

    @torch.no_grad()
    def inference(engine, batch):
        model.eval()
        images, targets = batch
        images = list(image.to(device) for image in images)
        outputs = model(images)
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        return {"y_pred": outputs, "y": targets, "x": [i.cpu() for i in images]}

    trainer = Engine(update_model)
    evaluator = Engine(inference)
    visualizer = Engine(inference)

    aim_logger = AimLogger(
        repo=os.path.join(log_dir, "aim"),
        experiment=experiment_name,
    )

    CocoMetric(convert_to_coco_api(val_dataset)).attach(evaluator, "mAP")

    @trainer.on(Events.EPOCH_COMPLETED)
    @one_rank_only()
    def log_validation_results(engine):
        evaluator.run(val_dataloader)
        visualizer.run(vis_dataloader)

    @trainer.on(Events.ITERATION_COMPLETED)
    def step_scheduler(engine):
        scheduler.step()
        aim_logger.log_metrics({"lr": scheduler.get_last_lr()[0]}, step=engine.state.iteration)

    @visualizer.on(Events.ITERATION_COMPLETED)
    def submit_vis_images(engine):
        aim_images = []
        outputs = engine.state.output
        for image, target, pred in zip(outputs["x"], outputs["y"], outputs["y_pred"]):
            image = (image * 255).byte()
            pred_labels = [Dataset.class2name[label.item()] for label in pred["labels"]]
            pred_boxes = pred["boxes"].long()
            image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

            target_labels = [Dataset.class2name[label.item()] for label in target["labels"]]
            target_boxes = target["boxes"].long()
            image = draw_bounding_boxes(image, target_boxes, target_labels, colors="green")

            aim_images.append(aim.Image(image.numpy().transpose((1, 2, 0))))
        aim_logger.experiment.track(aim_images, name="vis", step=trainer.state.epoch)

    losses = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg", "loss_average"]
    for loss_name in losses:
        RunningAverage(output_transform=lambda x: x[loss_name]).attach(trainer, loss_name)
    ProgressBar().attach(trainer, losses)
    ProgressBar().attach(evaluator)

    objects_to_checkpoint = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "scaler": scaler,
    }
    checkpoint = Checkpoint(
        to_save=objects_to_checkpoint,
        save_handler=DiskSaver(log_dir, require_empty=False),
        n_saved=3,
        score_name="mAP",
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)
    if resume_from:
        Checkpoint.load_objects(objects_to_checkpoint, torch.load(resume_from))

    aim_logger.log_params(
        {
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
        }
    )
    aim_logger.attach_output_handler(
        trainer, event_name=Events.ITERATION_COMPLETED, tag="train", output_transform=lambda loss: loss
    )
    aim_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="val",
        metric_names=["mAP"],
        global_step_transform=global_step_from_engine(trainer, Events.ITERATION_COMPLETED),
    )

    trainer.run(train_dataloader, max_epochs=epochs)


def main(
    experiment_name: str,
    gpus: Union[str, List[str], str] = "auto",
    nproc_per_node: Union[int, str] = "auto",
    dataset_root: str = "./dataset",
    log_dir: str = "./log",
    model: str = "fasterrcnn_resnet50_fpn",
    epochs: int = 13,
    batch_size: int = 4,
    lr: int = 0.01,
    download: bool = False,
    resume_from: str = None,
    visualize_images: int = 16,
) -> None:
    """
    Args:
        experiment_name: the name of each run
        dataset_root: dataset root directory for VOC2012 Dataset
        gpus: can be "auto", "none" or number of gpu device ids like "0,1"
        log_dir: where to put all the logs
        epochs: number of epochs to train
        model: model to use, possible options are
            "fasterrcnn_resnet50_fpn",
            "fasterrcnn_mobilenet_v3_large_fpn",
            "fasterrcnn_mobilenet_v3_large_320_fpn"
        batch_size: batch size
        lr: initial learning rate
        download: whether to automatically download dataset
        device: either cuda or cpu
        resume_from: path of checkpoint to resume from
        visualize_images: number of images to visualize each epoch
    """
    if model not in AVAILABLE_MODELS:
        raise RuntimeError(f"Invalid model name: {model}")

    if isinstance(gpus, int):
        gpus = (gpus,)
    if isinstance(gpus, tuple):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in gpus])
    elif gpus == "auto":
        gpus = tuple(range(torch.cuda.device_count()))
    elif gpus == "none":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        gpus = tuple()

    ngpu = len(gpus)

    backend = "nccl" if ngpu > 0 else "gloo"
    if nproc_per_node == "auto":
        nproc_per_node = ngpu if ngpu > 0 else max(multiprocessing.cpu_count() // 2, 1)

    # to prevent multiple downloads for prestrained checkpoint, create the model in the main process
    model = getattr(detection, model)(pretrained=True)

    if model.__class__.__name__ == "FasterRCNN":
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 21)
    elif model.__class__.__name__ == "RetinaNet":
        head = RetinaNetClassificationHead(
            model.backbone.out_channels, model.anchor_generator.num_anchors_per_location()[0], num_classes=21
        )
        model.head.classification_head = head

    with idist.Parallel(backend=backend, nproc_per_node=nproc_per_node) as parallel:
        parallel.run(
            run,
            "cuda" if ngpu > 0 else "cpu",
            experiment_name,
            dataset_root,
            log_dir,
            model,
            epochs,
            batch_size,
            lr,
            download,
            resume_from,
            visualize_images,
        )


if __name__ == "__main__":
    fire.Fire(main)
