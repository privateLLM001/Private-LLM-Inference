import transformers
import datasets
import evaluate
import logging
import numpy as np
import torchinfo
import os
import datetime
import torch

from dataclasses import dataclass, field
from typing import Optional

import modify_models

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

@dataclass
class RunArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    task_name: str = field(
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    output_dir: Optional[str] = field(
        default="./outputs/finetune"
    )
    epochs: Optional[int] = field(
        default=80
    )
    reg_loss_wgt: Optional[float] = field(
        default=0.1,
        metadata={"help": "Regularization Loss Weight"},
    )

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    parser = transformers.HfArgumentParser(RunArguments)
    run_args = parser.parse_args_into_dataclasses()[0]

    logger.setLevel(logging.INFO)
    max_seq_length = 512

    output_dir: str = run_args.output_dir
    if not output_dir.startswith("./outputs") and not output_dir.startswith("outputs"):
        output_dir = os.path.join("outputs", output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    fh = logging.FileHandler(os.path.join(output_dir, "log.txt"))
    fh.setLevel(logging.DEBUG)

    logging.root.addHandler(fh)
    

    task_name = run_args.task_name
    model_name = run_args.model_name #"prajjwal1/bert-tiny"

    epochs = run_args.epochs
    
    metric = evaluate.load("glue", task_name)

    
    training_args = transformers.TrainingArguments(
        do_train=True,
        do_eval=True,
        num_train_epochs=epochs,
        weight_decay=run_args.reg_loss_wgt,
        logging_steps=10,
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        per_device_train_batch_size=32,
        warmup_ratio=0.06,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
    )

    logger.info("Loading dataset ...")
    dataset = datasets.load_dataset("glue", task_name)
    
    label_list = dataset["train"].features["label"].names
    logger.info("Label list = {0}".format(label_list))

    num_labels = len(label_list)
    logger.info("Num labels = {0}".format(num_labels))

    config = transformers.AutoConfig.from_pretrained(
        model_name,
        num_labels = num_labels,
        finetuning_task = task_name,
    )
    logger.info("Config = {0}".format(config))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name
    )

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )

    torchinfo.summary(model, depth=100)

    is_regression = task_name == "stsb"

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != transformers.PretrainedConfig(num_labels=num_labels).label2id
        and task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )

    sentence1_key, sentence2_key = task_to_keys[task_name]
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=False, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result
    
    logger.info("Preprocessing the dataset")
    dataset = dataset.map(preprocess_function, batched=True)

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation_matched" if task_name == "mnli" else "validation"]

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: transformers.EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
    
    logger.info("Training args = {0}".format(training_args))

    logger.info("Initialize trainer")
    
    # Initialize our Trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=None,
    )

    train_result = trainer.train()

    torch.save(trainer.model.state_dict(), output_dir + "/model.pt")

    for entry in trainer.state.log_history:
        if "eval_loss" in entry:
            logger.warning(entry)

    def eval(model):
        eval_args = transformers.TrainingArguments(
            do_train=False,
            do_eval=True,
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_eval_batch_size=32,
        )
        trainer = transformers.Trainer(
            model=model,
            args=eval_args,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=None,
        )
        eval_result = trainer.evaluate()
        logger.warning(eval_result)
        return eval_result
    
    eval_result = eval(model)
    logger.warning("Evaluation result:")
    logger.warning(eval_result)
