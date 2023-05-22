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
    checkpoint_path: str = field()
    task_name: str = field(
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    replace_method: str = field()
    replace_layer: str = field()
    output_dir: Optional[str] = field(
        default="none"
    )
    single_layer_train_epochs: Optional[int] = field(
        default=10
    )
    layer_adapt_epochs: Optional[int] = field(
        default=10
    )
    layer_adapt_with_subsequent_layers: Optional[bool] = field(
        default=True,
    )
    finetune_epochs: Optional[int] = field(
        default=30
    )
    reg_loss_wgt: Optional[float] = field(
        default=0.0,
        metadata={"help": "Regularization Loss Weight"},
    )
    allowed_accuracy_drop: Optional[float] = field(
        default=0.01,
        metadata={"help": "Allowed Accuracy Drop"},
    )
    skip_train_single_layer: Optional[bool] = field(
        default=False,
    )
    evaluate_only: Optional[bool] = field(
        default=False,
    )
    hidden_state_bound: Optional[float] = field(
        default=16,
    )
    hidden_state_bound_loss_wgt: Optional[float] = field(
        default=0.05,
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1,
    )


class BoundControlTrainer(transformers.Trainer):

    def __init__(self, bound=16, alpha=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.bound = bound
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(output_hidden_states=True, return_dict=True, **inputs)
        logits = outputs.logits
        loss_fct = self.loss_fct(logits, labels)
        loss_bound = 0
        
        if self.alpha > 0:
            for hidden_state in outputs.hidden_states:
                # if hidden_state element absolute value > bound, then loss += l2norm(abs(element) - bound)
                norm = torch.norm(torch.max(torch.abs(hidden_state) - self.bound, torch.zeros_like(hidden_state)))
                # if norm > 100:
                #     print("Warn: norm = {0}, max = {1}".format(norm, torch.max(hidden_state)))
                loss_bound += norm

        loss = (1 - self.alpha) * loss_fct + self.alpha * loss_bound
        outputs = model(**inputs)
        return (loss, outputs) if return_outputs else loss


def load_previous_modification(file):
    if not os.path.exists(file):
        return []
    # load from a file, every line has 3 parts: layer_id, replace_layer, replace_method
    with open(file, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split(",") for line in lines]
    lines = [(int(line[0].strip()),line[1].strip(),line[2].strip()) for line in lines]
    return lines

def main():

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    max_seq_length = 512
    
    parser = transformers.HfArgumentParser(RunArguments)
    run_args = parser.parse_args_into_dataclasses()[0]

    task_name = run_args.task_name
    model_name = run_args.model_name #"prajjwal1/bert-tiny"
    checkpoint_path: str = run_args.checkpoint_path
    replace_method = run_args.replace_method
    replace_layer = run_args.replace_layer

    layer_adapt_epochs = run_args.layer_adapt_epochs
    single_layer_train_epochs = run_args.single_layer_train_epochs
    finetune_epochs = run_args.finetune_epochs
    allowed_accuracy_drop = run_args.allowed_accuracy_drop

    hidden_state_bound_loss_wgt = run_args.hidden_state_bound_loss_wgt
    hidden_state_bound = run_args.hidden_state_bound
    
    metric = evaluate.load("glue", task_name)

    output_dir: str = run_args.output_dir
    if output_dir == "none":
        output_dir = checkpoint_path.strip('/') + "-{0}-{1}".format(replace_layer, replace_method)
    if not output_dir.startswith("./outputs") and not output_dir.startswith("outputs"):
        output_dir = os.path.join("outputs", output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    fh = logging.FileHandler(os.path.join(output_dir, "log.txt"))
    fh.setLevel(logging.DEBUG)

    logging.root.addHandler(fh)
    
    logger.warning("Run args = {0}".format(run_args))

    model_file_name = "model.pt"

    logger.warning("Loading dataset ...")
    dataset = datasets.load_dataset("glue", task_name)
    
    label_list = dataset["train"].features["label"].names
    logger.warning("Label list = {0}".format(label_list))

    num_labels = len(label_list)
    logger.warning("Num labels = {0}".format(num_labels))

    config = transformers.AutoConfig.from_pretrained(
        model_name,
        num_labels = num_labels,
        finetuning_task = task_name,
    )
    logger.warning("Config = {0}".format(config))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name
    )

    def load_modified_model(checkpoint_path, model_file_name):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )
        # load checkpoint
        previous_modification = load_previous_modification(os.path.join(checkpoint_path, "modification.txt"))
        modify_models.modify_model_with_instruction(model, previous_modification)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, model_file_name)))
        return model, previous_modification

    model, previous_modification = load_modified_model(checkpoint_path, model_file_name)

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
            logger.warning(
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
    
    logger.warning("Preprocessing the dataset")
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

    def train(model, epochs, log_history=True):
        training_args = transformers.TrainingArguments(
            do_train=True,
            do_eval=True,
            num_train_epochs=epochs,
            weight_decay=run_args.reg_loss_wgt,
            logging_steps=10,
            output_dir=output_dir,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            per_device_train_batch_size=32//run_args.gradient_accumulation_steps,
            gradient_accumulation_steps=run_args.gradient_accumulation_steps,
            per_device_eval_batch_size=4,
            warmup_ratio=0.06,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            load_best_model_at_end=True,
        )
        trainer = BoundControlTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=None,
            alpha=hidden_state_bound_loss_wgt,
            bound=hidden_state_bound,
        )
        trainer.train()
        if log_history:
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
    
    logger.warning("\n\n\nEvaluating the original model (model.pt)")
    eval_result = eval(model)
    previous_eval_accuracy = eval_result["eval_accuracy"]
    logger.warn("original model eval accuracy: {0}".format(previous_eval_accuracy))

    if os.path.exists(os.path.join(checkpoint_path, "layerwise_finetuned_model.pt")):
        layerwise_model, _ = load_modified_model(checkpoint_path, "layerwise_finetuned_model.pt")
        logger.warning("\n\n\nEvaluating the layerwise finetuned model (layerwise_finetuned_model.pt)")
        eval_result = eval(layerwise_model)
        eval_accuracy = eval_result["eval_accuracy"]
        logger.warn("layerwise_finetuned_model eval accuracy: {0}".format(eval_accuracy))
        if eval_accuracy > previous_eval_accuracy:
            logger.warning("Using the layerwise finetuned model")
            model = layerwise_model
            previous_eval_accuracy = eval_accuracy

    if run_args.skip_train_single_layer == False:

        for layer_index in range(config.num_hidden_layers):

            logger.warning("\n\n\n---------------- Training layer {0} -------------------".format(layer_index))

            model, _ = load_modified_model(checkpoint_path, model_file_name)
            params = modify_models.modify_model_layer(model, layer_index, replace_layer, replace_method)
            if len(params) == 0:
                logger.warning("Nothing to train.")
                continue
            
            for name, param in model.named_parameters():
                param.requires_grad = False

            for param in params:
                param.requires_grad = True

            train(model, epochs=single_layer_train_epochs, log_history=False)
            eval_result = eval(model)
            eval_accuracy = eval_result["eval_accuracy"]
            logger.warning("Eval accuracy: {0}".format(eval_accuracy))

            # save parameters
            layer_trained_path = os.path.join(output_dir, "trained_layer_{0}.pt".format(layer_index))
            torch.save(params, layer_trained_path)

    # Finally get a whole model
    logger.warning("\n\n\n---------------------- Gradually merge trained single layers ---------------------")

    model, _ = load_modified_model(checkpoint_path, model_file_name)
    accepted_modification = []
    # save base model as accepted modified model
    torch.save(model.state_dict(), os.path.join(output_dir, "accepted_modified_model.pt"))

    eval_result = eval(model)
    logger.warning("Eval accuracy: {0}".format(eval_result["eval_accuracy"]))

    for layer_index in reversed(range(config.num_hidden_layers)):

        logger.warning("------- Replacing layer {0} -------".format(layer_index))
        params = modify_models.modify_model_layer(model, layer_index, replace_layer, replace_method)
        
        if len(params) > 0:
            layer_trained_path = os.path.join(output_dir, "trained_layer_{0}.pt".format(layer_index))
            if os.path.exists(layer_trained_path):
                logger.warning("Loading trained parameters for layer {0}".format(layer_index))
                trained_params = torch.load(layer_trained_path)
                for param, trained_param in zip(params, trained_params):
                    param.data = trained_param.data
            else:
                logger.warning("No trained parameters found for layer {0}".format(layer_index))
            
        for name, param in model.named_parameters():
            param.requires_grad = False

        if run_args.layer_adapt_with_subsequent_layers:
            for id in range(layer_index, config.num_hidden_layers):
                modify_models.set_trainable_layer(model, id)
        else:
            modify_models.set_trainable_layer(model, layer_index)

        if run_args.layer_adapt_with_subsequent_layers or layer_index == config.num_hidden_layers - 1:
            modify_models.set_trainable_classifier(model)

        train(model, epochs=layer_adapt_epochs, log_history=False)
        eval_result = eval(model)
        eval_accuracy = eval_result["eval_accuracy"]
        logger.warning("Eval accuracy: {0}".format(eval_accuracy))
        if eval_accuracy < previous_eval_accuracy - allowed_accuracy_drop:
            model = load_modified_model(checkpoint_path, model_file_name)[0]
            modify_models.modify_model_with_instruction(model, accepted_modification)
            model.load_state_dict(torch.load(os.path.join(output_dir, "accepted_modified_model.pt")))
            logger.warning("Replace {0} layer failed, reverting.".format(layer_index))
        else:
            accepted_modification.append((layer_index, replace_layer, replace_method))
            torch.save(model.state_dict(), os.path.join(output_dir, "accepted_modified_model.pt"))
            logger.warning("Replace {0} layer success".format(layer_index))
        
        logger.warning("Current accepted modification:")
        for each in accepted_modification:
            logger.warning("{0}".format(each))

    # Finally train the whole model
    
    logger.warning("\n\n\n---------------------- Training final merged modified model ---------------------")

    eval_result = eval(model)
    layerwise_eval_accuracy = eval_result["eval_accuracy"]
    logger.warning("Layerwise finetuned eval accuracy: {0}".format(layerwise_eval_accuracy))

    for name, param in model.named_parameters():
        param.requires_grad = True

    train(model, epochs=finetune_epochs, log_history=True)
    eval_result = eval(model)
    eval_accuracy = eval_result["eval_accuracy"]
    logger.warning("Final eval accuracy: {0}".format(eval_accuracy))

    save_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), save_path)

    # save the accepted modification to file
    accepted_modification = previous_modification + accepted_modification
    with open(os.path.join(output_dir, "modification.txt"), "w") as f:
        for each in accepted_modification:
            f.write("{0},{1},{2}\n".format(each[0], each[1], each[2]))

    if layerwise_eval_accuracy > eval_accuracy:
        logger.warning("Using the layerwise finetuned model ({0})".format(layerwise_eval_accuracy))
        # remove model.pt
        os.remove(save_path)
        # rename layerwise_finetuned_model.pt to model.pt
        os.rename(os.path.join(output_dir, "accepted_modified_model.pt"), save_path)
    else:
        # remove layerwise_finetuned_model.pt
        os.remove(os.path.join(output_dir, "accepted_modified_model.pt"))

if __name__ == "__main__":
    main()






