import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    BertForSequenceClassification, 
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import get_peft_config, get_peft_model, PeftType, TaskType, PeftConfig, LoraConfig, PeftModel

from transformers import Trainer
from transformers.trainer import _is_peft_model

import torch
import os
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score, precision_recall_fscore_support
from src.logger import logger

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, loss_fn=None, num_labels: int = 6, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights 
        
        pos_weight = class_weights if class_weights is not None else torch.ones(num_labels)
        if torch.cuda.is_available():
            pos_weight = pos_weight.to("cuda")
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fn(logits, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # if self.args.average_tokens_across_devices and (self.model_accepts_loss_kwargs or self.compute_loss_func):
        #     loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

class LoRAFinetuner:
    def __init__(
        self,
        model_name="intfloat/multilingual-e5-base",
        num_labels=6,
        lora_rank=8,
        lora_alpha=16, #rank*2
        lora_dropout=0.1,
        output_dir="models/lora_model_1",
        data_path=None,          
        dataframe=None,          
        validation_data_path=None,  
        validation_dataframe=None  ,
        target_modules=["query", "key", "value"],
        modules_to_save =['classifier'],
        classifier_only: bool = False,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.output_dir = output_dir
        self.data_path = data_path
        self.dataframe = dataframe
        self.validation_data_path = validation_data_path
        self.validation_dataframe = validation_dataframe

        self.target_modules = target_modules
        self.modules_to_save = modules_to_save

        self.classifier_only = classifier_only

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            trust_remote_code=True
        )

        logger.info(str(self.model))

        
        if classifier_only:
            self._freeze_all_except_classifier()
        else:
            self._add_lora()

    
    def _freeze_all_except_classifier(self):
        """Замораживает все слои кроме головы классификатора"""
        for param in self.model.parameters():
            param.requires_grad = False
        
        classifier_unfrozen = False
        for name, param in self.model.named_parameters():
            if 'classifier' in name.lower():
                param.requires_grad = True
                classifier_unfrozen = True
                logger.info(f"✅ Разморожен слой классификатора: {name}")
        
        
        if not classifier_unfrozen:
            raise ValueError("Не удалось найти слои классификатора для разморозки!")
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"\n📊 Статистика параметров:")
        logger.info(f"   Всего параметров: {total_params:,}")
        logger.info(f"   Обучаемых параметров: {trainable_params:,} ({trainable_params/total_params:.2%})")
        
        logger.info("\n🔍 Обучаемые слои:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.info(f"   - {name} (размер: {tuple(param.shape)})")

    def _tokenize_function(self, examples, prompt: str = "query:"):
        text = examples['token']
        text_sents =  text.split(".")
        title = text_sents[0]
        annotation = ".".join(text_sents[1:]) if len(text_sents) > 1 else ""

        text_to_tokenize = prompt + title + "[SEP]" + annotation + "[SEP]"

        tokenized_data = self.tokenizer(
        text_to_tokenize,
        truncation=True,
        padding='max_length',
        max_length=512
        )

        raw_label = examples['label']

        label_vector = torch.zeros(self.num_labels, dtype=torch.float)
        if isinstance(raw_label, str):
            indices = list(map(int, raw_label.split(',')))
        elif isinstance(raw_label, list):
            indices = raw_label
        elif isinstance(raw_label, int):
            indices = [raw_label]
        else:
                raise ValueError(f"Label must be string like '0,2,5' or list like [0, 2, 5], got {type(raw_label)}: {raw_label}")
        label_vector[indices] = 1.0 

        tokenized_data["labels"] = label_vector.numpy().astype(np.float32)

        return tokenized_data
    
    def _prepare_dataset(self):
        """Загрузка и предобработка данных"""
        
        if self.dataframe is not None:
            df = self.dataframe
        elif self.data_path is not None:
            if self.data_path.endswith(".csv"):
                df = pd.read_csv(self.data_path)
            elif self.data_path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(self.data_path)
            else:
                raise ValueError("Unsupported file format. Use CSV/Excel.")
        else:
            raise ValueError("Training data not provided. Specify 'data_path' or 'dataframe'.")

        assert "token" in df.columns and "label" in df.columns, (
            "Training data must have 'token' (text) and 'label' (class) columns"
        )

        if self.validation_dataframe is not None:
            val_df = self.validation_dataframe
        elif self.validation_data_path is not None:
            if self.validation_data_path.endswith(".csv"):
                val_df = pd.read_csv(self.validation_data_path)
            elif self.validation_data_path.endswith((".xlsx", ".xls")):
                val_df = pd.read_excel(self.validation_data_path)
            else:
                raise ValueError("Validation data must be CSV/Excel.")
        else:
            df, val_df = train_test_split(
                df,
                test_size=0.2,
                stratify=df["label"] if "label" in df.columns else None
            )

        if val_df is not None:
            assert "token" in val_df.columns and "label" in val_df.columns, (
                "Validation data must have 'token' (text) and 'label' (class) columns"
            )

        train_dataset = Dataset.from_pandas(df, preserve_index=False)
        eval_dataset = Dataset.from_pandas(val_df, preserve_index=False) if val_df is not None else None

        class_frequencies = df['label'].value_counts().sort_index().to_dict()
        logger.info(f"Частота классов: {class_frequencies}")

        total_samples = sum(class_frequencies.values())
        class_weights = {cls: total_samples / (len(class_frequencies) * freq) for cls, freq in class_frequencies.items()}
        self.class_weights_tensor = torch.tensor([class_weights[cls] for cls in sorted(class_weights.keys())], dtype=torch.float32)
        logger.info(f"Веса классов: {self.class_weights_tensor}")
        

        train_dataset = train_dataset.map(self._tokenize_function, batched=False)
        eval_dataset = eval_dataset.map(self._tokenize_function, batched=False)

        return train_dataset, eval_dataset


    def _add_lora(self):
        peft_config = LoraConfig(
                                  task_type=TaskType.SEQ_CLS,
                                  r=self.lora_rank,
                                  target_modules=self.target_modules,
                                  modules_to_save=self.modules_to_save,
                                  lora_alpha=self.lora_alpha,
                                  lora_dropout=self.lora_dropout,
                                  )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def train(
        self,
        batch_size=4,
        epochs=10,
        learning_rate=5e-4,
        early_stopping_patience=3,
        early_stopping_threshold=0.0,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
    ):
        
        def custom_collate(batch):
            input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
            attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
            labels = torch.tensor([item["labels"] for item in batch], dtype=torch.float32)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
    

        train_dataset, eval_dataset = self._prepare_dataset()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            weight_decay=0.00001,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=True,
            label_names=["labels"],
            dataloader_drop_last=True,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["tensorboard"], 
            logging_dir=os.path.join(self.output_dir, "tb_logs"),
            logging_strategy="epoch",
            logging_steps=1,  
        )

        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=custom_collate,
            class_weights=self.class_weights_tensor,
            num_labels=self.num_labels,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience,
                                            early_stopping_threshold=early_stopping_threshold)]

        )
        trainer.train()
        self.model.save_pretrained(self.output_dir)

    def load(self, lora_model, base_model='intfloat/multilingual-e5-base', num_labels=46):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = PeftConfig.from_pretrained(lora_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels, torch_dtype=torch.float16)
        model.to(device)
        self.model = PeftModel.from_pretrained(model, lora_model, num_labels=num_labels)
        self.model.to(device)
        

    def predict(self, texts, batch_size=8, threshold=0.5):
        """
        Метод для выполнения предсказаний с использованием обученной модели и LoRA для Multi-Label Classification.
        :param texts: Список текстов для классификации.
        :param batch_size: Размер батча для обработки.
        :param threshold: Порог для определения активности класса (по умолчанию 0.5).
        :return: Список списков/массивов, где каждый внутренний элемент представляет one-hot вектор
                 предсказанных активных классов для соответствующего текста.
                 Например: [[0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1]]
        """
        self.model.eval()

        all_predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            tokenized_inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )

            device = next(self.model.parameters()).device
            tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}

            with torch.no_grad():
                outputs = self.model(**tokenized_inputs)
                logits = outputs.logits

            probabilities = torch.sigmoid(logits)
            binary_predictions = (probabilities >= threshold).float()
            binary_predictions_np = binary_predictions.cpu().numpy()

            for pred in binary_predictions_np:
                all_predictions.append(pred.astype(int).tolist())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_predictions

    def evaluate_model(self, test_data_path, batch_size=8, threshold=0.5):
        """
        Метод для оценки модели на тестовом наборе данных.

        Args:
            test_data_path (str): Путь к файлу .csv с тестовыми данными.
                                   Должен содержать колонки 'token' (текст) и 'label' (метки в формате "0,2,5" или [0, 2, 5]).
            batch_size (int): Размер батча для инференса. По умолчанию 8.
            threshold (float): Порог для определения активности класса. По умолчанию 0.5.

        Returns:
            dict: Словарь с вычисленными метриками.
        """
        if test_data_path.endswith(".csv"):
            df = pd.read_csv(test_data_path)
        else:
            raise ValueError("Test data must be a CSV file.")

        assert "token" in df.columns and "label" in df.columns, (
            f"Test data must have 'token' (text) and 'label' (class) columns. "
            f"Available columns: {list(df.columns)}"
        )


        test_dataset = Dataset.from_pandas(df, preserve_index=False)
        test_dataset = test_dataset.map(self._tokenize_function, batched=False)

        y_true_list = [item['labels'] for item in test_dataset]
        y_true = torch.tensor(y_true_list, dtype=torch.float) 

        texts = df['token'].tolist()
        y_pred_lists = self.predict(texts=texts, batch_size=batch_size, threshold=threshold)
        y_pred = torch.tensor(y_pred_lists, dtype=torch.float)

        accuracy = accuracy_score(y_true, y_pred)

        hamming = hamming_loss(y_true, y_pred)


        jaccard_micro = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
        jaccard_macro = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
        jaccard_samples = jaccard_score(y_true, y_pred, average='samples', zero_division=0)

        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

        metrics = {
            "accuracy": accuracy,
            "hamming_loss": hamming,
            "jaccard_micro": jaccard_micro,
            "jaccard_macro": jaccard_macro,
            "jaccard_samples": jaccard_samples,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
        }

        return metrics