
user_message="""
I have a text dataset. It contains 690 utterances from TV shows such as The Big Bang Theory and Friends. 
Each sample contains the following available information:
- utterance: the target utterance to classify
- sarcasm: the binary label, where 1 / true means sarcastic and 0 / false means non-sarcastic
- speaker: the speaker of the target utterance
- context: the previous dialogue context of the target utterance
- context_speakers: the speakers of each sentence in the context

Augmentations must modify the dataset and increase the datasize to 2~4 times larger.

The selected model is BERT+SVM. 
The three target metrics are "f1_weighted“ in five folds cross-validation ,the multiplier of dataset size after augmentation and the augmentation processing time.

Please generate several augmentation pipelines to improve validation performance. If there is a trade-off between metrics, prioritize average validation f1-score across the 5 folds.
The augmentation could be applied on utterance only, context only, or both.
The augmentation should operate only at the raw text level of training dataset before BERT feature extraction.

"""

train_spec="""

The original dataset is a JSON file with 690 samples. Each sample may contain the following fields:
- id
- utterance
- speaker
- context
- context_speakers
- sarcasm

The field sarcasm is the binary label. Do not use any information other than utterance, speaker, context, context_speakers, and sarcasm during model training or evaluation. In particular, do not use the show field even if it exists in the JSON.

A sample looks like this:
{
    "1_60": {
        "utterance": "It's just a privilege to watch your mind at work.",
        "speaker": "SHELDON",
        "context": [
            "I never would have identified the fingerprints of string theory in the aftermath of the Big Bang.",
            "My apologies. What's your plan?"
        ],
        "context_speakers": [
            "LEONARD",
            "SHELDON"
        ],
        "show": "BBT",
        "sarcasm": true
    }
}

The desired text variables for each sample are:

context_text:
LEONARD: I never would have identified the fingerprints of string theory in the aftermath of the Big Bang.
SHELDON: My apologies. What's your plan?

utterance_text:
SHELDON: It's just a privilege to watch your mind at work.

Important text construction requirement:
- Do NOT add literal section labels such as "context:" or "utterance:" into the text passed to BERT.
- The variable context_text should contain only the speaker-prefixed context sentences.
- The variable utterance_text should contain only the speaker-prefixed target utterance.
- Keep context_text and utterance_text as two separate variables.
- Do NOT concatenate them into one string before tokenization.
- Feed them into BERT as sentence-pair input:

  tokenizer(batch_contexts, batch_utterances, ...)

- Therefore the actual BERT input should be:

  [CLS] context_text [SEP] utterance_text [SEP]

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

Baseline requirement:
- Implement a baseline-only training script.
- Do not perform any data augmentation.
- Do not modify the original JSON file.
- Do not modify split_indices.p.
- Tokenizer and sequence-length requirements:
- Use MAX_LENGTH = 256 for every baseline and augmentation experiment.
- Tokenize the two text variables using exactly:
- The augmentation should return complete augmented training dataset, including every original training sample.
- The augmentation must not modify/change the original dataset.
  tokenizer(
      batch_contexts,
      batch_utterances,
      padding=True,
      truncation=True,
      max_length=256,
      return_tensors="pt",
  )

BERT requirement:
- Use the pretrained HuggingFace model bert-base-uncased as the feature extractor.
- Use AutoTokenizer.from_pretrained("bert-base-uncased").
- Use AutoModel.from_pretrained("bert-base-uncased").
- Do not implement a custom BERT-like model.
- Do not use a randomly initialized fallback model.
- If bert-base-uncased cannot be loaded, raise an explicit error.
- Do not fine-tune BERT.
- Put BERT in eval mode.
- Use torch.no_grad() during feature extraction.

BERT feature extraction requirement:
- Obtain hidden states from BERT by setting output_hidden_states=True.
- Use the [CLS] token representation.
- Average the [CLS] representations from the last 4 hidden layers.
- The embedding for each sample should be:

  mean([
      hidden_states[-1][:, 0, :],
      hidden_states[-2][:, 0, :],
      hidden_states[-3][:, 0, :],
      hidden_states[-4][:, 0, :]
  ])

- Do not use mean pooling over all tokens.

Classifier requirement:
- Use SVM as the classifier.
- Specifically use an RBF SVM.
- Use StandardScaler before SVM.
- Use sklearn Pipeline so that scaling is fitted only on the training fold.
- Use a fixed SVM hyperparameter C=10 for all baseline and augmentation experiments.
- Do not perform GridSearchCV or any per-pipeline hyperparameter tuning.
- Keep all SVM hyperparameters identical across the baseline and all augmentation pipelines.




Final result:
- Report weighted-F1 multiplier of dataset size after augmentation and data augmentation processing time across the 5 folds.

"""