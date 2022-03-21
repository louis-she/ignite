# FasterRCNN Example with Ignite

In this example, we show how to use _Ignite_ to train a image detection model with PyTorch built-in Faster RCNN implementation.

- Using VOC2012 datasets (~2GB).
- Vlidate with COCO mAP.
- Visualization.
- Aim as experiment manager.

## Requirements:

- pytorch-ignite: `pip install pytorch-ignite`
- [torchvision](https://github.com/pytorch/vision): `pip install torchvision`
- [aim](https://github.com/aimhubio/aim): `pip install aim`

Alternatively, install the all requirements using `pip install -r requirements.txt`.

## Usage:

```bash
python faster_rcnn.py
```

For details on accepted arguments:

```bash
python faster_rcnn.py --help
```

The datasets will be downloaded automatically, if you already has the VOC2012, you can use the `--dataset-root` option to locate it.
