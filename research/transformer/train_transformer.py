from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer



def group_texts(examples):
    block_size = 128
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def main():
    eli5 = load_dataset("eli5", split="train_asks[:5000]")
    eli5 = eli5.train_test_split(test_size=0.2)
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    eli5 = eli5.flatten()

    def preprocess_function(examples):
        """
        cut input if size is to long
        """
        return tokenizer([" ".join(x) for x in examples["answers.text"]], truncation=True)

    tokenized_eli5 = eli5.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=eli5["train"].column_names,
    )

    lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # train
    model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == '__main__':
    main()
