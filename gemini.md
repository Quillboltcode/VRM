

## **Context: Research Project – Adaptive Recursive CNN for FER**

You are an expert PyTorch Researcher. You are building a novel **Recursive Convolutional Network with Adaptive Computation Time (ACT)** for Facial Expression Recognition (RAF-DB/AffectNet).

**Core Hypothesis:** Harder facial expressions (e.g., Contempt, Fear) require more processing steps than easier ones (e.g., Happy). We use a recursive loop to "think" longer on hard samples.

---

## **Technical Specifications (The "Improved" Architecture)**

### **1. Model Architecture (`model.py`)**

You must implement `RecursiveFER` with the following strict logic to avoid "spatial lobotomy" and "broken gradients":

* **Input Stem:** 2 Conv layers + GroupNorm + GELU to project .
* **Recursive Block (The Loop):**
* **Input:** 4D Tensor . **DO NOT FLATTEN.**
* **Operation:** .
* *Note:* Maintain spatial dimensions throughout the recursion.


* **Halting Mechanism (Differentiable ACT):**
* **Halting Head:** GlobalAvgPool  Linear  Sigmoid.
* **Logic:**
* Calculate  at step .
* Calculate .
* Accumulate Output: .
* Accumulate Cost: .
* Update Remaining: .





### **2. Loss Function (`loss.py`)**

Implement `DifferentiablePonderLoss`:

* **Equation:** .
* **Class-Aware Weighting:**
* Allow passing `class_weights` to the Ponder Loss.
* *Logic:* Multiply the ponder cost by a factor based on the target class (e.g., penalize "Happy" heavily if it takes long, be lenient on "Surprise").



### **3. Data Pipeline (`dataset.py`)**

* **Dataset:** RAF-DB (7 classes).
* **Augmentation (Albumentations):**
* ShiftScaleRotate, HorizontalFlip, CoarseDropout (Cutout), ColorJitter.
* *Critical:* FER datasets are small; heavy augmentation prevents overfitting.


* **Kaggle Optimization:**
* If running on Kaggle, unzip the dataset from `/kaggle/input` to `/kaggle/working` **before** initializing the DataLoader to avoid I/O bottlenecks.



---

## **Implementation Plan**

### **Step 1: Setup & Data (`dataset.py`, `utils.py`)**

* Create a `RafDBDataset` class that parses the `train_label.txt` file.
* Implement a helper function `extract_dataset()` for Kaggle.
* **Action:** Write the dataset class with Albumentations support.

### **Step 2: The Model (`model.py`)**

* Implement the `RecursiveFER` class using the **Graves ACT accumulation logic**.
* **Constraint:** Ensure `forward` returns `(logits, ponder_cost, step_probs_list)`.
* **Action:** Write the model code focusing on the `for` loop logic to ensure gradients flow back to the Halting Head.

### **Step 3: Training Loop (`train.py`)**

* Use `wandb` to log:
* `Train/Loss`, `Train/Accuracy`.
* `ACT/Average_Steps`: The mean of the ponder cost.
* `ACT/Steps_Per_Class`: (Optional) Log average steps broken down by emotion.


* **Optimization:** Use `torch.optim.AdamW` with Cosine Annealing scheduler.

### **Step 4: Evaluation & Visualization**

* Create a script to run inference on the Test Set.
* **Plot:** A histogram showing the distribution of "Steps Taken" for the whole dataset.
* **Plot:** A "Confusion Matrix of Steps" (e.g., X-axis = Emotion, Y-axis = Avg Steps).

---

## **Coding Constraints**

1. **Type Hints:** Use `Tensor`, `List`, `Tuple` typing everywhere.
2. **Modularity:** Keep the `RecursiveBlock` separate from the main `RecursiveFER` class.
3. **Stability:** Initialize the Halting Head bias to a slightly positive value (e.g., `+1.0`) to encourage the model to output higher halting probabilities initially (prevents "thinking forever" at the start).

---

## **Kaggle Specific Instructions**

* Assume input data is at `/kaggle/input/raf-db-dataset/`.
* Save all checkpoints to `/kaggle/working/checkpoints/`.
* Use `kaggle_secrets` to retrieve the `WANDB_API_KEY`.

