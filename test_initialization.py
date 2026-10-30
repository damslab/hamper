from Llm_Client.claude_code_client import ClaudeCodeChatClient
from Llm_Client.codex_client import CodexClient
from User_Application.engine import build_engine_input
from Downstream_ML.engine import MLEngine
from Augmentation_Engine.engine import AugEngine
from Coding_Engine.engine import PipelineCodeEngine
from Feedback_Engine.engine import FeedbackEngine
from Feature_Extraction_Engine.engine import FeatureEngine
import json
from pathlib import Path

#-----------------Training Script Specification------------------------------------------------

train_spec = """
Write a complete, runnable Python training script for image classification
on an ImageNet100-style dataset using ResNet50. Implement one fixed
no-augmentation baseline with a reserved augmentation insertion point.

Dataset:
- Root: /home/xiaoma/.cache/kagglehub/datasets/ambityga/imagenet100/versions/8
- Training folders: train.X1, train.X2, train.X3, train.X4.
- Validation folder: val.X.
- Each split contains WNID class subfolders.
- Scan all split folders, sort the detected class names, and build one
  shared class_to_idx mapping.
- Sort image paths and use all images in their existing splits.

Model:
- Use torchvision.models.resnet50(weights=None, num_classes=num_classes).
- Train all parameters from scratch.
- Use model.train() during training and model.eval() during validation.

Preprocessing and augmentation position:
- The CPU Dataset reads images, converts them to RGB, resizes the shorter
  side to 256 using bilinear interpolation with antialiasing, and applies
  a deterministic center crop to 224x224.
- Return uint8 tensors of shape [3, 224, 224] and integer class indices.
- After device transfer, convert images to float32 and divide by 255.
- Reserve an augmentation insertion point here, before normalization.
  In the baseline, this position leaves images and targets unchanged.
- Do not implement any augmentation operations.
- Normalize exactly once on the training device using:
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  - Before calling the augmentation function, convert integer training
  labels to float32 one-hot targets of shape [B, num_classes].
  Augmenttion pipeline receive:
  Input images: float32 [B, 3, 224, 224] in [0, 1], before normalization.
  Input targets: float32 [B, num_classes], never integer class indices.
- The augmentation may return soft targets of the same shape.
  Pass these targets directly to soft-target cross-entropy without
  argmax, integer conversion, or conversion back to one-hot.
- When no augmentation, leave images and one-hot targets unchanged.
- Keep validation labels as integer class indices.
- Validation uses the same deterministic preprocessing and normalization,
  without an augmentation insertion point.

Training:
- Train for exactly 30 epochs without early stopping.
- Use SGD with momentum=0.9, weight_decay=1e-4,
  dampening=0, and nesterov=False.
- Apply weight decay to all trainable parameters.
- Use batch_size=64 for both training and validation.
- Use accumulation_steps=2.
- Normalize accumulated gradients by the actual number of samples in
  each accumulation group, including the final incomplete group.
- Perform one optimizer update per group and clear gradients between groups.
- Convert training labels to float32 one-hot targets.
- Compute soft-target cross-entropy in float32:
      loss = -(targets * log_softmax(logits.float(), dim=1)).sum(dim=1).mean()
- Do not use label smoothing, gradient clipping, model EMA, or additional
  training techniques.

Learning rate:
- Epochs 1-3: use 0.01, 0.055, and 0.1 respectively.
- Epochs 4-15: use 0.1.
- Epochs 16-23: use 0.01.
- Epochs 24-30: use 0.001.
- Set the learning rate at the beginning of each epoch.
- Do not scale learning rates with batch size.

Device and precision:
- Use one CUDA device if available; otherwise use CPU.
- On CUDA, use float16 torch.amp.autocast and torch.amp.GradScaler.
- Use channels_last for the model and model inputs on CUDA.
- Use non_blocking=True for device transfers.
- On CPU, use float32 without AMP or gradient scaling.

DataLoader:
- Use num_workers=min(8, os.cpu_count() or 0).
- Set pin_memory=True on CUDA and False on CPU.
- Enable persistent_workers when num_workers > 0.
- Use prefetch_factor=2 when num_workers > 0; otherwise omit it.
- Shuffle training data using a seeded torch.Generator.
- Do not shuffle validation data.
- Use drop_last=False for both loaders.

Reproducibility:
- Use seed=42 and implement seed_all(seed).
- Seed Python random, NumPy, torch, and CUDA.
- Seed Python random and NumPy in DataLoader workers using
  torch.initial_seed() % (2**32).
- Seed the training DataLoader generator with seed.
- Set torch.backends.cudnn.deterministic=True.
- Set torch.backends.cudnn.benchmark=False.
- Do not reset seeds inside epoch or batch loops.

Evaluation and outputs:
- Validate the complete validation split after every epoch using
  torch.inference_mode().
- Use original integer labels and standard cross-entropy for validation.
- Aggregate losses by sample count.
- Compute top-1 accuracy as total correct predictions / total samples,
  reported in [0, 1].
- Number epochs from 1 to 30.
- Measure epoch time over training and validation, excluding log and
  checkpoint writing. Synchronize CUDA at timing boundaries.
- Training script should optput the target mectrics :best validation accuracy and average epoch time.

"""
#----------------------------Task Prompt-------------------------------------------------------
user_message = """
ImageNet-100 is a subset of the ImageNet-1k dataset from the ImageNet
Large Scale Visual Recognition Challenge 2012. It contains 100 randomly
selected classes, as specified in the Labels.json file.

The training split contains 1300 images per class, and the validation
split contains 50 images per class.

Task:
Design a data-augmentation pipeline for training a ResNet-50 image
classifier from scratch.

Training environment:
- Model: ResNet-50, trained from scratch without pretrained weights.
- Training duration: 30 epochs.
- Optimizer: SGD with momentum.
- Training batch size: 64.
- Input image resolution: 224 x 224.
- CUDA is used when available, with mixed-precision training.

Optimization objectives:
1. Maximize best validation accuracy.
2. Minimize average epoch time.

Augmentation context:
- The preprocessing pipeline already performs image resizing and
  deterministic center cropping.
- Augmentation is applied to mini-batches on the training device,
  after preprocessing but before normalization.
- Input images are float32 tensors of shape [B, 3, 224, 224],
  with pixel values in [0, 1].
- Input targets are float32 one-hot tensors of shape [B, 100].
- The augmentation pipeline may modify both images and targets.
- Soft targets are supported through soft-target cross-entropy.
- The augmentation pipeline must return images and targets with
  compatible shapes and types.

Design considerations:
- Explore augmentation strategies that may improve generalization
  under the fixed training budget.
- Consider the trade-off between augmentation strength, training
  convergence, and computational overhead.
- Both image-only and image-target joint augmentations are allowed.
- Avoid unnecessary preprocessing operations already performed
  by the fixed training pipeline.
- Do not modify the model architecture, optimizer, learning-rate
  schedule, or other fixed training configurations.

Implementation constraints:
1. Augmentation must be applied during training only.
2. Augmentation must operate at batch level.
3. Augmentation must be placed after resize and center crop,
   but before normalization.
4. Validation and test-time data must not be augmented.
5. The pipeline must preserve compatibility with the fixed
   training interface.
"""

#---------------------------Working path setup-----------------------------

workdir=Path("/home/xiaoma/Agent_hamper/Downstream_ML").resolve()
dataset_path='/home/xiaoma/.cache/kagglehub/datasets/ambityga/imagenet100/versions/8'
output_path = Path("pipeline_results.jsonl")
output_hypothesis_path = Path("round_hypothesis.jsonl")
output_roundanalysis_path = Path("round_analysis.jsonl")

# Codex and Claude Code require a valid Git repository as their working directory.
# The specified workdir must point to a directory initialized with Git (git init).
system_client=CodexClient(workdir="/home/xiaoma/Empty",model="gpt-5.6-luna")
system_client.start_new_task(
    reset_usage=True,
)
downsteam_client=CodexClient(workdir="/home/xiaoma/Empty",model="gpt-5.6-sol")
downsteam_client.start_new_task(
    reset_usage=True,
)

#Generate task specification configuration file
spec=build_engine_input(system_client,user_message)

#Generate and save training script
downstream_ml_engine=MLEngine(workdir,downsteam_client,spec,dataset_path,output_path,train_spec)
result = downstream_ml_engine.generate_augmentation_train_script()
if not isinstance(result, dict):
    raise TypeError(f"call_codex() should return a dict, got {type(result).__name__}")

code = result.get("code")
if not isinstance(code, str) or not code.strip():
    raise ValueError("Codex returned invalid or empty 'code' field")

code_path = workdir / "generated_train_pipeline.py"
code_path.write_text(code, encoding="utf-8")
print(f"Python code saved to: {code_path}")

merged = spec | {k: v for k, v in result.items() if k != "code"}
result_no_code = {k: v for k, v in result.items() if k != "code"}
merged = spec | result_no_code
save_dir = Path("/home/xiaoma/Agent_hamper")
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "merged.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

# downstream_ml_engine.create_result_file(output_path)
# downstream_ml_engine.create_result_file(output_hypothesis_path)
# downstream_ml_engine.create_result_file(output_roundanalysis_path)

#Traing Baseline
# baseline_result = downstream_ml_engine.train(None)
# baseline_result["pipeline_description"] = "No augmentation baseline"
# baseline_result["family"]="No augmentation baseline"
# baseline_result["arm_role"]="canonical"
# baseline_result["code"] = None
# baseline_result["status"] = "success"
# baseline_result["round"]=0
# baseline_result['pipeline_id']=0
# baseline_result['stochastic']=False
# downstream_ml_engine.append_result(baseline_result)