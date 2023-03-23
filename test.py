from deepspeed.profiling.flops_profiler import FlopsProfiler
from transformers import Trainer, OPTForSequenceClassification, TrainingArguments, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset

raw_data = load_dataset("glue", "mrpc")
ckpt = 'facebook/opt-1.3b'
tokenizer = AutoTokenizer.from_pretrained(ckpt)
def tokenize_function(examples):
  return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
tokenized_datasets = raw_data.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer)

training_args = TrainingArguments("config.json")
model=OPTForSequenceClassification.from_pretrained(ckpt)
num_labels = len(model.config.id2label)
model=OPTForSequenceClassification.from_pretrained(ckpt, num_labels=num_labels)
prof = FlopsProfiler(model)

profile_step = 5
print_profile= True
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()
'''
for step, batch in enumerate(data_loader):
  # start profiling at training step "profile_step"
  if step == profile_step:
    prof.start_profile()

  # forward() method
  loss = model(batch)

  # end profiling and print output
  if step == profile_step: # if using multi nodes, check global_rank == 0 as well
    prof.stop_profile()
    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()
    if print_profile:
        prof.print_model_profile(profile_step=profile_step)
    prof.end_profile()

  # runs backpropagation
  loss.backward()

  # weight update
  optimizer.step()
'''
