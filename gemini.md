

# gemini.md

## **Context: Research Project – Adaptive Recursive CNN for FER**

You are an expert Computer Vision Researcher and PyTorch Engineer. You are helping me build a research codebase for a specific novel architecture: **Recursive Convolutional Networks with Adaptive Computation Time (ACT)** applied to Facial Expression Recognition (FER).

### **The Core Idea**

Standard CNNs (like ResNet) use fixed depth. We hypothesize that some facial expressions are "easy" (e.g., clearly Happy) and need shallow processing, while others are "ambiguous" (e.g., Contempt vs. Neutral) and need deeper, iterative refinement.

* **Architecture:** A "Tiny" recursive backbone that loops a single block  times.
* **Mechanism:** A **Halting Gate** (sigmoidal output) that decides when to stop processing for each image.
* **Datasets:** RAF-DB (Real-world Affective Faces) and AffectNet.

---

## **Technical Specification**

### **1. Dependencies**

The codebase must utilize:

* `torch`, `torchvision` (Latest stable)
* `albumentations` (For heavy data augmentation: ShiftScaleRotate, Cutout, ColorJitter)
* `timm` (For referencing baseline architectures if needed)
* `wandb` (For logging accuracy vs. average steps taken)
* `scikit-learn` (For confusion matrices and weighted F1-score)

### **2. The Recursive Model Architecture (`model.py`)**

You must implement a class `RecursiveFER` with:

* **Embedding Layer:** A small standard CNN stem (e.g., 2 Conv layers) to project the image () into a hidden state  (e.g., ).
* **Recursive Block:** A single `nn.Module` that takes state  and outputs .
* *Components:* Conv3x3 -> GroupNorm -> GELU -> Conv3x3.
* *Memory:* A GRUCell or a Residual Link ().


* **Halting Unit (ACT):**
* A small dense layer that looks at  (pooled) and outputs a probability .
* **Logic:** Stop when cumulative probability  or when .


* **Classifier Head:** A final linear layer mapping the final state to Class Logits (7 classes for RAF-DB).

### **3. Loss Function (`loss.py`)**

The loss must be a combination of:

1. **Classification Loss:** CrossEntropy on the final prediction.
2. **Ponder Cost:** A regularization term to penalize thinking too long.
* 
* Where  is the "remainder" or average steps taken.


3. **Deep Supervision (Optional):** You may implement an option to calculate loss at *every* step to stabilize training.

### **4. Data Pipeline (`dataset.py`)**

* **RAF-DB:** An `ImagefolderDataset` with only train folder and test folder with **7 Basic emotions**
* **Balancing:** The training set is imbalanced. Implement `WeightedRandomSampler` or class weighting in the Loss function.

### **5. Experiment Tracking (`train.py`)**

* We need to track **Accuracy** and **Average Steps** simultaneously.
* **Proof of Concept Goal:** Show that "Easy" images (High confidence) use fewer steps than "Hard" images.

---

## **Task Instructions**

When I ask you to generate code, follow these rules:

1. **Modular Code:** Do not dump one giant file. Separate `model.py`, `dataset.py`, `train.py`.
2. **Type Hinting:** Use `def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:`
3. **ACT Implementation:** Be very careful with the Adaptive Computation Time logic. It is tricky to batch. You can implement the "Naïve" version first (masking out finished samples in the batch).
4. **Visualize:** Provide a snippet to visualize which images took 1 step vs. 10 steps.
Provide histogram for **Average step per class** for RAFDB dataset.

---

## **Plan of Action**

1. **Setup:** Create the file structure and using `uv add` or `uv pip install` for dependencies.
2. **Data:** Write the `RafDBDataset` loader.
3. **Model:** Implement the `RecursiveBlock` and `HaltingMechanism`.
4. **Train:** Write the training loop with Ponder Loss.
5. **Analyze:** Create a script to run inference and plot "Steps Taken" histograms.

---

## **Appendix: Kaggle Environment Constraints**

**IMPORTANT:** We are running this on Kaggle. Adapt the code accordingly:

1. **Data Loading:**
* Assume the dataset is a `ImagefodlerDataset` with file located at `/kaggle/input/rafdb/`.
* Set `num_workers=4` (Kaggle has 2-4 CPU cores usually; setting it too high crashes).
* Split data using GroupKFold for testing with seed for repo reproducibility.


2. **Output & Logging:**
* All outputs must go to `/kaggle/working/`.
* Integrate `wandb` for logging Loss, Accuracy, and **Average Recursion Depth** per epoch.
* Code must retrieve the WandB API key using `kaggle_secrets`.


3. **Memory Management:**
* The P100 GPU has 16GB VRAM.
* Since we are unrolling the loop  times, memory usage grows linearly with .
* If OOM (Out of Memory) occurs, instruct to reduce `batch_size` or implement **Gradient Checkpointing** (`torch.utils.checkpoint`) on the Recursive Block.



