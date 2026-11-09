train_spec = """
The original dataset is a JSON file with 690 samples. Each sample may contain the following fields:
- id
- utterance
- speaker
- context
- context_speakers
- sarcasm
- show

The field sarcasm is the binary label. Do not use the show field during model training or evaluation.

This task uses only the audio file corresponding to each target utterance.

Do not use:
- utterance text
- context text
- speaker identity as a model feature
- context speaker identity as a model feature
- show field
- video features
- any non-audio information as model input features

The label field is:
- sarcasm

Audio file location:
- All audio files are already utterance-level audio files.
- All utterance audio files are stored in:
  /home/xiaoma/Augmentation_Test/MustARD/data/audio_wav_by_id
- Each audio file is named by the sample id / JSON key.
- The audio filename format is:
  {sample_id}.wav
- Example:
  sample id: 1_60
  audio path:
  /home/xiaoma/Augmentation_Test/MustARD/data/audio_wav_by_id/1_60.wav

Audio input requirement:
- Use only the .wav file corresponding to each sample id.
- Each .wav file already contains only the target utterance audio.
- For each JSON sample, construct the audio path as:
  audio_path = "/home/xiaoma/Augmentation_Test/MustARD/data/audio_wav_by_id/{sample_id}.wav"
- sample_id must be the original JSON key, such as "1_60".
- Do not infer audio paths from utterance text, speaker, show, or context.
- If an audio file cannot be found, raise a clear error instead of silently skipping the sample.
- Do not modify any original audio files.

Official split requirement:
For training and evaluation, only use the exact split definition in:
/home/xiaoma/Augmentation_Test/MustARD/data/split_indices.p

This file defines the official 5-fold cross-validation split. Follow it exactly.

Important split and dataset order requirements:
- Do not change the order of samples before using split_indices.p.
- Do not sort JSON keys.
- Preserve the original JSON item order exactly as loaded.
- The split_indices.p file may have been created in Python 2 or an older Python environment.
- When loading split_indices.p in Python 3, use a compatibility-safe loading strategy:
  first try pickle.load(f), and if UnicodeDecodeError occurs, retry with pickle.load(f, encoding="latin1").

Feature output requirement:
- The returned features must be compatible with an sklearn RBF SVM classifier.
- Prefer fixed-dimensional numeric features.
- If feature_fn returns pandas DataFrame, numpy array, scipy sparse matrix, or torch tensor, the training script may safely convert it for sklearn compatibility.
- The training script should verify that feature row counts match the corresponding labels.

Classifier requirement:
- Use SVM as the classifier.
- Specifically use an RBF SVM.
- Use StandardScaler before SVM.
- Use sklearn Pipeline so that scaling is applied before SVM.
- Do not use GridSearchCV.
- Do not tune SVM hyperparameters.
- Use fixed RBF SVM parameters:
  - C = 1.600
  - gamma = 0.0065
- Implement the classifier as:
  SVC(kernel="rbf", C=1.600, gamma=0.0065)

Feature extraction placement:
- If feature_fn is stateless, deterministic, does not use labels, and does not fit dataset-level state, the training script should call feature_fn once on the full ordered audio_paths list before cross-validation:
  X_all = feature_fn(audio_paths)
- Then, for each official fold, construct:
  X_train = X_all[train_indices]
  X_eval = X_all[eval_indices]
- This avoids repeated audio feature extraction across folds while preserving the official split.
- StandardScaler must still be inside the sklearn Pipeline and fit only on X_train within each fold.
- If feature_fn requires fitted state, learned parameters, vocabularies, projections, encoders, or dataset-level statistics, do not precompute on all samples. In that case, fit state on the training fold only and transform the evaluation fold safely.
- For this initial feature-search system, prefer stateless per-sample feature_fn implementations.

Cross-validation requirement:
- Use the official 5 folds from split_indices.p.
- For each outer fold:
  - Use the official train indices as the training set.
  - Use the official validation/test indices as the evaluation set according to the split file.
  - Prepare train_audio_paths and eval_audio_paths from the sample ids.
  - Train the sklearn Pipeline on X_train and y_train.
  - Evaluate on X_eval and y_eval.
- Do not evaluate on training data.

Final result:
- Report weighted-F1 across the 5 official folds.
- The target metric is f1_weighted.
"""
user_message = """
I have the audio part of the MUSTARD dataset. It contains 690 utterance-level audio samples from TV shows such as The Big Bang Theory and Friends.

Each audio file is already the target utterance audio and is named by sample id.

Each sample has a binary sarcasm label:
- sarcasm: 1 / true means sarcastic
- sarcasm: 0 / false means non-sarcastic

I want to perform feature extraction only on these utterance-level audio files.

The selected model is RBF SVM with fixed parameters:
- C = 1.6
- gamma = 0.0065
Choose MFCC-20 as the baseline.
The target metric is f1_weighted across the official 5-fold cross-validation split.

Please generate several feature extraction strategies for utterance-level audio that may improve average validation weighted-F1 across the 5 folds.

The feature extraction strategies should be compatible with sklearn RBF SVM, so prefer fixed-dimensional numeric features."""
dataset_path='/home/xiaoma/Augmentation_Test/MustARD/data/sarcasm_data.json'