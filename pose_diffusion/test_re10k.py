import hydra
import torch

from accelerate import Accelerator
from datasets.real_estate_10k_dataloader import RealEstate10KDataset
from functools import partial
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch3d.renderer.cameras import PerspectiveCameras
from test import prefix_with_module
from util.geometry_guided_sampling import geometry_guided_sampling
from util.match_extraction import extract_match
from util.metric import camera_to_rel_deg, calculate_auc_np
from util.train_util import set_seed_and_print

import numpy as np


def load_existing_checkpoint(checkpoint_path, model, accelerator):
    checkpoint = torch.load(checkpoint_path)
    try:
        model.load_state_dict(prefix_with_module(checkpoint), strict=True)
    except:
        model.load_state_dict(checkpoint, strict=True)

    accelerator.print(f"Resumed training from checkpoint: {checkpoint_path}")


def generate_model_conditioning_function(config, image_paths, crop_params, image_shape):
    if config.GGS.enable:
        kp1, kp2, i12 = extract_match(image_paths=image_paths, image_info=crop_params)

        if kp1 is not None:
            correspondence_data = {
                "kp1": kp1,
                "kp2": kp2,
                "i12": i12,
                "img_shape": tuple(image_shape)
            }

            config.GGS.pose_encoding_type = config.MODEL.pose_encoding_type
            geometric_sampling_config = OmegaConf.to_container(config.GGS)
            return partial(
                geometry_guided_sampling,
                matches_dict=correspondence_data,
                GGS_cfg=geometric_sampling_config
            )


def compute_sequence_error(sequence_name, re10k_dataset, model, config, accelerator):
    sequence_image_count = len(re10k_dataset.sequence_metadata_map[sequence_name])
    random_image_ids = np.random.choice(
        sequence_image_count,
        min(sequence_image_count, config.test.num_frames),
        replace=False
    )

    batch, image_paths = re10k_dataset.get_data(sequence_name, random_image_ids)

    translations = batch["T"].to(accelerator.device)
    rotations = batch["R"].to(accelerator.device)
    focal_lengths = batch["fl"].to(accelerator.device)
    principal_points = batch["pp"].to(accelerator.device)

    perspective_cameras = PerspectiveCameras(
        focal_length=focal_lengths.reshape(-1, 2),
        principal_point=principal_points.reshape(-1, 2),
        R=rotations.reshape(-1, 3, 3),
        T=translations.reshape(-1, 3),
        device=accelerator.device
    )

    images = batch["images"].to(accelerator.device).unsqueeze(0)
    model_conditioning_function = generate_model_conditioning_function(
        config,
        image_paths,
        batch["crop_params"],
        batch["images"].shape
    )

    with torch.no_grad():
        predictions = model(
            images,
            cond_fn=model_conditioning_function,
            cond_start_step=config.GGS.start_step,
            training=False
        )["pred_cameras"]

    rotation_error_deg, translation_error_deg = camera_to_rel_deg(
        predictions,
        perspective_cameras,
        accelerator.device,
        len(images)
    )

    print(f"\tPair Rotation Error (Deg): {rotation_error_deg.mean():10.2f}")
    print(f"\tPair Translation Error (Deg): {translation_error_deg.mean():10.2f}")

    return rotation_error_deg.cpu().numpy(),translation_error_deg.cpu().numpy()


@hydra.main(config_path="../cfgs/", config_name="re10k_test")
def test_model_re10k(config: DictConfig):
    OmegaConf.set_struct(config, False)

    accelerator = Accelerator(even_batches=False, device_placement=False)
    accelerator.print("Model Config:", OmegaConf.to_yaml(config), accelerator.state)

    torch.backends.cudnn.benchmark = config.test.cudnnbenchmark if not config.debug else False

    if config.debug:
        accelerator.print(f"{'*' * 10} Debug Mode {'*' * 10}")
        torch.backends.cudnn.deterministic = True

    set_seed_and_print(config.seed)

    model = accelerator.prepare(instantiate(config.MODEL, _recursive_=False).to(accelerator.device))

    if checkpoint_path := config.test.resume_ckpt:
        load_existing_checkpoint(checkpoint_path, model, accelerator)

    re10k_dataset = RealEstate10KDataset(
        split="test",
        dataset_images_path=config.test.images_path,
        dataset_metadata_path=config.test.metadata_path
    )

    accuracy_upper_bounds = [5, 15, 30]
    avg_rotation_errors = np.zeros(len(accuracy_upper_bounds))
    avg_translation_errors = np.zeros(len(accuracy_upper_bounds))
    avg_auc_30_error = 0

    for sequence_name in re10k_dataset.sequence_names:
        rotation_errors, translation_errors = (
            compute_sequence_error(sequence_name, re10k_dataset, model, config, accelerator))

        for i, upper_bound in enumerate(accuracy_upper_bounds):
            avg_rotation_errors[i] += np.mean(rotation_errors < upper_bound) * 100
            avg_translation_errors[i] += np.mean(translation_errors < upper_bound) * 100

        avg_auc_30_error += calculate_auc_np(rotation_errors, translation_errors, max_threshold=30) * 100

    num_sequences = len(re10k_dataset.sequence_names)
    avg_rotation_errors /= num_sequences
    avg_translation_errors /= num_sequences
    avg_auc_30_error /= num_sequences

    for i, upper_bound in enumerate(accuracy_upper_bounds):
        print(f"Average Rotation Error @ {upper_bound}: {avg_rotation_errors[i]}")
        print(f"Average Translation Error @ {upper_bound}: {avg_translation_errors[i]}")

    print(f"Average AUC 30 Error: {avg_auc_30_error}")


if __name__ == '__main__':
    test_model_re10k()
