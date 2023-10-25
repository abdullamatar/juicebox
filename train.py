from pathlib import Path
from anomalib.models import Padim
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
)
from anomalib.utils.callbacks.export import ExportCallback, ExportMode
from anomalib.data.utils import read_image
from anomalib.deploy import TorchInferencer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from anomalib.data.folder import Folder

from anomalib.data.task_type import TaskType

# data_dir = Path.cwd()
data_dir = Path.cwd() / "Reels"
# print(data_dir)

data_module = Folder(
    root=data_dir,
    normal_dir="normal",
    abnormal_dir="abnormal",
    normal_split_ratio=0.1,
    image_size=(256, 256),
    train_batch_size=64,
    eval_batch_size=64,
    task=TaskType.CLASSIFICATION,
    normalization="imagenet",
    seed=44,
)
data_module.setup()
data_module.prepare_data()
i, data = next(enumerate(data_module.val_dataloader()))
print(data.keys())
print(type(data["image"]))
print(
    len(data_module.test_data), len(data_module.train_data), len(data_module.val_data)
)
# data_module.train_data[0]

model = Padim(
    input_size=(256, 256),
    backbone="resnet18",
    layers=["layer1", "layer2", "layer3", "layer4"],
)


callbacks = [
    MetricsConfigurationCallback(
        task=TaskType.CLASSIFICATION,
        image_metrics=["AUROC"],
    ),
    ModelCheckpoint(
        every_n_epochs=10,
        monitor="image_AUROC",
    ),
    PostProcessingConfigurationCallback(
        normalization_method=NormalizationMethod.MIN_MAX,
        threshold_method=ThresholdMethod.ADAPTIVE,
    ),
    MinMaxNormalizationCallback(),
    ExportCallback(
        input_size=(256, 256),
        dirpath=str(Path.cwd()),
        filename="4layer_model.pt",
        export_mode=ExportMode.TORCH,
    ),
]
trainer = Trainer(
    callbacks=callbacks,
    accelerator="auto",
    auto_scale_batch_size=False,
    check_val_every_n_epoch=4,
    devices=1,
    gpus=None,
    max_epochs=20,
    num_sanity_val_steps=0,
    val_check_interval=1.0,
)
trainer.fit(model=model, datamodule=data_module)
