# ClearML Remote Agent

## BERT Sentiment Classification with PyTorch Lightning

A simple, self-contained example of training a BERT model for sentiment classification using PyTorch Lightning on the IMDB dataset.

### Features
- 🤗 Uses `bert-base-uncased` from Hugging Face
- ⚡ PyTorch Lightning for clean, organized code
- 📊 IMDB dataset for sentiment analysis (positive/negative)
- 📈 Automatic metrics tracking (accuracy, F1 score)
- 💾 Model checkpointing and early stopping
- 🔬 ClearML integration for experiment tracking and remote execution
- 🔐 Secure credential management via environment variables

### Quick Start

1. **Create a virtual environment:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up ClearML credentials:**

   a. Create a ClearML account at https://app.clear.ml (if you don't have one)
   
   b. Get your API credentials:
      - Go to https://app.clear.ml
      - Click on your profile (top right) → Settings
      - Go to "Workspace" tab
      - Click "Create new credentials" or copy existing ones
      - You'll get an `Access Key` and `Secret Key`
   
   c. Create a `.env` file in the project root:
   ```bash
   touch .env
   ```
   
   d. Add your ClearML credentials to `.env`:
   ```bash
   CLEARML_API_ACCESS_KEY=your_access_key_here
   CLEARML_API_SECRET_KEY=your_secret_key_here
   ```
   
   **Important:** Never commit the `.env` file to git (it's already in `.gitignore`)

4. **Run training:**
```bash
python train.py
```

5. **View results:**
   - The training will automatically log all metrics to ClearML
   - Open https://app.clear.ml to view:
     - Real-time training metrics (loss, accuracy, F1)
     - Hyperparameters
     - Console logs
     - System metrics (GPU/CPU usage)
     - Model artifacts

6. **Deactivate virtual environment (when done):**
```bash
deactivate
```

### What the Script Does

- Loads ClearML credentials securely from `.env` file
- Initializes ClearML Task for experiment tracking
- Downloads the IMDB dataset automatically
- Uses a subset of 1000 training samples for quick demonstration
- Fine-tunes BERT for binary sentiment classification
- Trains for up to 3 epochs with early stopping
- Saves the best model checkpoint based on validation accuracy
- Logs all metrics to ClearML (training/validation loss, accuracy, and F1 score)
- Tracks hyperparameters and system resources in ClearML

### Configuration

You can modify these parameters in `train.py`:

```python
BATCH_SIZE = 16          # Batch size for training
MAX_EPOCHS = 3           # Maximum number of epochs
LEARNING_RATE = 2e-5     # Learning rate for AdamW optimizer
MAX_LENGTH = 128         # Maximum sequence length for BERT
SUBSET_SIZE = 1000       # Number of samples to use (set to None for full dataset)
```

### Output

The script will:
- Create a `checkpoints/` directory with the best model
- Log all training progress to ClearML in real-time
- Save the best model based on validation accuracy
- Generate a ClearML Task with full experiment tracking

You can view all results in the ClearML web interface at https://app.clear.ml including:
- Training curves and metrics
- Hyperparameters
- Console logs
- System metrics (GPU/CPU/Memory usage)
- Model artifacts and checkpoints

### Hardware Requirements

- **CPU**: Works fine, but slower
- **GPU**: Recommended for faster training (automatically detected)
- **Memory**: ~4GB RAM minimum

### Full Dataset Training

To train on the full IMDB dataset (25k samples), modify `train.py`:

```python
SUBSET_SIZE = None  # Use full dataset
MAX_EPOCHS = 3      # May need more epochs
```

Note: Full dataset training will take significantly longer!

### Troubleshooting

**Error: "ClearML credentials not found"**
- Make sure your `.env` file exists in the project root
- Check that `CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY` are set correctly
- Ensure there are no extra spaces or quotes around the keys

**Error: "Connection to ClearML failed"**
- Verify your API credentials are correct
- Check your internet connection
- Make sure you can access https://app.clear.ml in your browser

**Training is slow**
- GPU is recommended for faster training
- Reduce `BATCH_SIZE` or `SUBSET_SIZE` if you have limited memory
- The script automatically detects and uses GPU if available

---

Репозиторий к статье на DeepSchool. Использование clearml agent для обучения на colab
