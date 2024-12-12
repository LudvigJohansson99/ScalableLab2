**Lab 2: Fine-Tuning a Pre-Trained Language Model**

## **1. Introduction**

This Lab focuses on fine-tuning a pre-trained language model using the **SFTTrainer** and **Transformers library**. The fine-tuned model will be periodically saved as checkpoints to ensure persistence. This allows the training process to be resumed at a later time without loss of progress.

---

## **2. Objectives**

- Fine-tune a pre-trained model using **SFTTrainer**.
- Save model checkpoints.
- Learn to manage model persistence to prevent loss of training progress.

---

## **3. Key Changes Made**

1. **Checkpoint Storage**:

   - Checkpoints are saved in `/content/drive/MyDrive/Checkpoints/`.
   - The directory is created using `os.makedirs()` to ensure it exists before training.

2. **Trainer Customization**:

   - `output_dir` was updated to `/content/drive/MyDrive/Checkpoints/`.
   - Set `save_steps=25` to save checkpoints every 25 steps.
   - Set `save_total_limit=3` to retain only the **latest 3 checkpoints**, removing older ones to save space.

---

## **4. Environment Setup**

### **1️⃣ Mount Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
```

This mounts Google Drive to the Colab file system so that files can be saved persistently to Google Drive, even after the Colab session is disconnected.

### **2️⃣ Set Up Checkpoint Directory**

```python
import os
output_dir = "/content/drive/MyDrive/Checkpoints"
os.makedirs(output_dir, exist_ok=True)
print(f"Checkpoints will be saved in: {output_dir}")
```

This creates the directory `/content/drive/MyDrive/Checkpoints` to store all the model checkpoints.

---

## **5. Training the Model**

### **3️⃣ Trainer Configuration**

```python
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc = 2,
    packing = False,  
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = True,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,  # Save to Google Drive
        save_steps = 25,  # Save a checkpoint every 25 steps
        save_total_limit = 3  # Keep only 3 latest checkpoints
    ),
)
```

---

## **6. Key Parameters Explanation**

| **Parameter**                       | **Description**                       | **Value**                             |
| ----------------------------------- | ------------------------------------- | ------------------------------------- |
| **output\_dir**                     | Path to save checkpoints              | `/content/drive/MyDrive/Checkpoints/` |
| **save\_steps**                     | Save a checkpoint every N steps       | `25`                                  |
| **save\_total\_limit**              | Limit the total number of checkpoints | `3`                                   |
| **num\_train\_epochs**              | Number of training epochs             | `1`                                   |
| **fp16**                            | Use mixed-precision training (faster) | `True`                                |
| **learning\_rate**                  | Learning rate for training            | `2e-4`                                |
| **per\_device\_train\_batch\_size** | Batch size per GPU device             | `2`                                   |

---

## **8. Troubleshooting**

| **Issue**                    | **Possible Cause**             | **Solution**                           |
| ---------------------------- | ------------------------------ | -------------------------------------- |
| **Checkpoints not saving**   | Google Drive not mounted       | Run `drive.mount('/content/drive')`    |
| **Path not found error**     | Checkpoint path doesn't exist  | Run `os.makedirs(output_dir)`          |
| **Training restarts from 0** | Checkpoint not loaded properly | Use `from_pretrained(checkpoint_path)` |
| **Insufficient Drive Space** | Too many checkpoints saved     | Set `save_total_limit=3`               |

---

## **9. File Structure of Saved Checkpoints**

```
/content/drive/MyDrive/Checkpoints/
  ├── checkpoint-25/
  │   ├── config.json
  │   ├── pytorch_model.bin
  │   ├── optimizer.pt
  │   └── scheduler.pt
  ├── checkpoint-50/
  │   ├── config.json
  │   ├── pytorch_model.bin
  │   ├── optimizer.pt
  │   └── scheduler.pt
  └── checkpoint-75/
      ├── config.json
      ├── pytorch_model.bin
      ├── optimizer.pt
      └── scheduler.pt
```

---

## **10. Conclusion**

This Lab demonstrates how to fine-tune a model using **SFTTrainer** and store model checkpoints. This method ensures that model progress is not lost, even if the Colab runtime disconnects. By using **TrainingArguments** to control checkpoint frequency, the training process becomes more efficient, and by leveraging Google Drive, we ensure model persistence across sessions.

---

## Model-centric approach
Tune Hyperparameters:
Experiment with learning rates, batch sizes, and weight decay values. Use a grid search to find the best hyperparameters. See what optimizes convergence.
Scale the number of layers, attention heads, or hidden dimensions if computational resources permit.
Try larger pre-trained models
Experiment with dropout rates.


## Data-centric approach
Identify New Data Sources:
Use synthetic data generation techniques such as chatGPT to create additional training examples.
Introduce variability in the training data through techniques such as paraphrasing or synonyms
Ensure the dataset covers a wide variety of examples to reduce model bias
