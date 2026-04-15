# Dissertation

run notebook = jupyter notebook

before running fresh, delete processed folder
run the notebook entries sequentially
wasnt pushing to github
increased epochs 2-4-6
Threshold search - 0.15 through 0.60
changed from trainer to weighted trainer

# Notes

- included extra PII patterns (email, phone, student number, names, organisation names).
- Training was too shallow at first, so I increased epochs from 2 to 4, then to 6.
- Added more useful metrics (micro and macro F1, plus precision and recall) so results are easier to interpret.
- Added threshold tuning after training, then simplified it to one global threshold so the notebook stays readable.
- Constrained threshold search to 0.15 to 0.60 to reduce overly aggressive positive predictions.
- Added validation and test positive prediction rate checks to understand precision and recall tradeoffs.
- Switched from default trainer loss to class-weighted loss using per-label pos_weight.
- Kept saving model artifacts and thresholds every run so results are reproducible.
- Git pushes were taking too long so added .gitignore to stop uploading the large model files
- code wasn't uploading to github due to the size of the model files
- first run of inferance had a lot of escalations so the code needs tweaking

### Preprocessing notes

- Confirmed required columns and handled missing text values before training.
- Kept text cleaning and masking outputs in processed files to separate preprocessing from modelling.
- Built the multi-label target columns and taxonomy file for consistent label mapping.
- Generated train, validation, and test split files so training runs are repeatable.
- Used the cleaned text field only for tokenization to avoid mixing raw and processed inputs.

## Plan for preprocessing

- Step 1: foundations - environment setup + schema validation.✔️
- Step 2: normalisation - PII masking + text cleaning✔️
- Step 3: target prep - Multi label taxonomy + label encoding
- Step 4: reproducable data splits - train/ validate/ test splitting

- Load the split files.
- Tokenize text_clean with the DistilBERT tokenizer.
- Build PyTorch datasets/dataloaders.
- Fine-tune the classifier for multi-label output.
- Then do evaluation
- Compute F1, precision, recall, and per-label metrics.
- Check urgent classes separately if that matters for your dissertation.
- add explainability and escalation
- LIME for local explanations.
- Confidence thresholds for low-confidence flags.
- Optional human-in-the-loop decision logic.
- make inference file

## sources

- pandas read_csv: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
- Python re module: https://docs.python.org/3/library/re.html
- spaCy named entity recognition: https://spacy.io/usage/linguistic-features#named-entities
- pandas groupby: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
- pandas merge: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html
- scikit-learn MultiLabelBinarizer (conceptual reference): https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html
- Hugging Face Trainer: https://huggingface.co/docs/transformers/main_classes/trainer
- Hugging Face TrainingArguments: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
- DistilBERT model docs: https://huggingface.co/docs/transformers/model_doc/distilbert
- scikit-learn f1_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- scikit-learn precision_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
- scikit-learn recall_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
- scikit-learn classification threshold tuning: https://scikit-learn.org/stable/modules/classification_threshold.html
- https://www.youtube.com/watch?v=8yrD0hR8OY8&t=61s
- https://www.youtube.com/watch?v=ZvsH09XGuZ0 