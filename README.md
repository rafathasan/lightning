# 🚀 PyTorch Lightning Template with WandB Integration

A clean and structured template for deep learning projects using PyTorch Lightning ⚡, configured via Lightning CLI 🎛️, and integrated with Weights & Biases 📊 for experiment tracking.

---

## 📂 Project Structure

```bash
.
├── configs/            # 🛠️ Configuration files (YAML/Hydra)
├── data/               # 🗃️ Dataset storage and preprocessed data
├── models/             # 🤖 Model architectures and saved checkpoints
├── .gitignore          # 🙈 Specify files/folders to ignore
├── README.md           # 📖 You're here!
├── main.py             # 🧠 Main training/evaluation script
├── requirements.txt    # 📦 Dependency list
└── run.sh              # 🚀 Bash script to launch training
```

---

## 🏁 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up WandB**  
   - Create a free account at [wandb.ai](https://wandb.ai)
   - Add your API key to environment variables:
     ```bash
     export WANDB_API_KEY=your_api_key_here
     ```

---

## ⚙️ Configuration with Lightning CLI

Configure hyperparameters, model architectures, and data settings via YAML files in `configs/`. Example config:

```yaml
# configs/train_config.yaml
seed: 42
batch_size: 32
max_epochs: 100
optimizer: adamw
```

Modify training parameters using:
```bash
python main.py --config configs/your_config.yaml
```

---

## 🔮 Weights & Biases Integration

Automatically track:
- Metrics (loss/accuracy)
- Hyperparameters
- Model gradients/parameters
- Hardware utilization

![WandB Dashboard](https://storage.googleapis.com/wandb-production.appspot.com/lavanyashukla/images/projects/37229/5622b3bd.png?Expires=1739480413&GoogleAccessId=gorilla-files-url-signer-man%40wandb-production.iam.gserviceaccount.com&Signature=eZqjB5OB8OqvlfBCTiM9bBOwmU9%2B1YTFlNcvIkVUxoX2CHJzeeHjzzKcxie7PRrrMsqzloQjvi9ypON9qzOqCQOQfiEGIu8dd8HX2xXyOZU%2FTUfNMtpcg5S84Hg3QVVcac9LgOzyB%2BvBXbv5KJg5DWEw4fr1OUEfy%2FGCcF9qAWW5S1Qx7%2FAqXmrTI7MtaJXIP6t8z4cxrc5WiUeBv%2BGFlAqzg5K8kdvY49hMtQzUsBQzALneHKH3lJ3XfwjXOc4Upci3ccEtymbLHeCanvfY%2FnPvb8vvCca3N8jNn8dttXFwgED8tZpzT0ggfdRcI2SUVkucYJ8wsA0P8noIr39ZxA%3D%3D)  
*Example WandB dashboard output*

---

## 🧪 Example Usage

**Train model:**
```bash
./run.sh  # Uses default config from configs/
```

**Custom training:**
```bash
python main.py \
    --trainer.max_epochs=50 \
    --data.batch_size=64 \
    --model.learning_rate=0.001 \
    --logger=wandb
```

**Run inference:**
```bash
python main.py --mode test --ckpt_path models/your_model.ckpt
```

---

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

Made with ❤️ by [rafathasan](www.github.com/rafathasan). Powered by ⚡ PyTorch Lightning and 🧠 Weights & Biases.
