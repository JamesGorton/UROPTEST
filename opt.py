from deepspeed.profiling.flops_profiler import FlopsProfiler
from transformers import AdamW, AutoModelForCausalLM, TrainingArguments, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader
from ignite.utils import manual_seed
from ignite.contrib.handlers import PiecewiseLinear
manual_seed(42)

ckpt = "facebook/opt-1.3b"
model = AutoModelForCausalLM.from_pretrained(ckpt, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(ckpt)

raw_data = load_dataset("Abirate/english_quotes")
data = raw_data.map(lambda samples: tokenizer(samples['quote']), batched=True)

train_dataset = data["train"].shuffle()

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)

model=AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-4)
num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = PiecewiseLinear(
        optimizer, param_name="lr", milestones_values=milestones_values
    )
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

prof = FlopsProfiler(model)

profile_step = 1
print_profile= True

def train_step(engine, batch):  
    model.train()
    prof.start_profile()
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    prof.stop_profile()
    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()
    if print_profile:
        prof.print_model_profile(profile_step=profile_step)
    prof.end_profile()
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return loss
    
from ignite.engine import Engine
trainer = Engine(train_step)
from ignite.engine import Events
trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)
from ignite.contrib.handlers import ProgressBar
pbar = ProgressBar()
pbar.attach(trainer, output_transform=lambda x: {'loss': x})

def evaluate_step(engine, batch):
    model.eval()

    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    return {'y_pred': predictions, 'y': batch["labels"]}
    
train_evaluator = Engine(evaluate_step)
validation_evaluator = Engine(evaluate_step)
#metric= load_metric("accuracy")
from ignite.metrics import Accuracy

Accuracy().attach(train_evaluator, 'accuracy')
Accuracy().attach(validation_evaluator, 'accuracy')
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    train_evaluator.run(train_dataloader)
    metrics = train_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    print(f"Training Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f}")
    
def log_validation_results(engine):
    validation_evaluator.run(eval_dataloader)
    metrics = validation_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    print(f"Validation Results - Epoch: {engine.state.epoch}  Avg accuracy: {avg_accuracy:.3f}")

trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)
trainer.run(train_dataloader, max_epochs=num_epochs)

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
