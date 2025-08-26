# Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals in Impulsive Noise-Dominated Channels

This repository contains the official implementation used to perform the experiments and generate the results presented in our paper:
**“Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals in Impulsive Noise-Dominated Channels.”**

---

## Repository Structure

```
main
├── config/          # Configuration files for training and evaluation
├── core/            # Custom IterableDataset, training engine, and utility functions
├── data/            # Data generation functions and classes
├── models/          # Model definitions (Masked Symbol Model and classification head)
├── inter.py         # Inference script for impaired signals
├── infer_clean.py   # Inference script for clean (non-impaired) signals
├── plot_metrics.py  # Script to plot metrics across experiment runs
├── requirements.txt # Python dependencies
└── train.py         # Training script
```

---

## Running the Code

1. **Configure the experiment**
   Modify the configuration files under `config/` as needed.

2. **Select the data generation objective**
   Open `data_generator.py` and locate the comment starting with `OVERRIDE`.
   Uncomment the corresponding line based on your experimental objective.

3. **Train the model**

   ```bash
   python train.py
   ```

4. **Run inference**

   * For impaired signals:

     ```bash
     python inter.py
     ```
   * For clean signals:

     ```bash
     python infer_clean.py
     ```

5. **Plot the metrics**

   ```bash
   python plot_metrics.py
   ```