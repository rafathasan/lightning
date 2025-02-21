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
1. **Use the Template**

   - Click the `Use this template` > `Create a new repository`
   - Enter `Repository name` > `Create repository`
   - Click the `Code` > `Codespaces` > `Create codespace on main`
   
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up WandB**  
   - Create a free account at [wandb.ai](https://wandb.ai)
   - Add your API key in `run.sh`:
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

![WandB Dashboard](https://docs-beta.wandb.ai/assets/images/quickstart_image-c06aed771fcdd2c38d5abdd76ff77b36.png)
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
