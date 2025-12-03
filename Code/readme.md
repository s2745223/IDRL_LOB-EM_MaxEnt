# Code — IDRL-LOB

This folder contains the full implementation of the **Inverse Delayed Reinforcement Learning (IDRL)** framework used in the paper _“Estimating Strategic Delays in Limit Order Books via EM–MaxEnt IRL.”_

The code is organised into three main entry points and a sequence of internal pipeline modules.

---

## 📌 Main Scripts

### **1. `main.py` — Preprocessing**
This script handles:
- tick-level LOB data ingestion  
- feature engineering  
- state–action construction  
- delay-window generation

### **2. `main2.py` — EM–MaxEnt IRL**
This script runs the complete EM–MaxEnt pipeline:
- probabilistic state–action responsibilities  
- delay-window likelihood updates  
- reward parameter updates  
- mixture-of-experts gating (multi-strategy model)  
- convergence checks

### **3. `main3.py` — Results & Plots**
This script generates:
- responsibility heatmaps  
- delay distribution plots  
- strategy-separation plots  
- time-delay evolution  
- behavioural interpretation figures

---

## 🔧 Internal Pipeline Stages

These modules define the internal steps executed by the main scripts.  
They are listed here in the rough order they are executed:

1. **`step1_ingest.py`** — Load raw LOB/tick data  
2. **`step2_action_representation.py`** — Construct action sequence  
3. **`step3_state_features.py`** — Build state vectors  
4. **`step4_state_sanitize.py`** — Clean/validate states  
5. **`step5_delay_windows.py`** — Generate delay brackets  
6. **`step6_feature_map.py`** — Build feature maps for MaxEnt  
7. **`step8_joint_responsibilities.py`** — E-step responsibilities  
8. **`step9_delay_prior.py`** — Prior distribution over delays  
9. **`step10_update_psi_gating.py`** — Update gating weights (multi-strategy)  
10. **`step10_update_theta_local_maxent.py`** — Update local MaxEnt reward params  
11. **`step11_convergence.py`** — EM convergence logic  
12. **`step12_save_thetas.py`** — Export learned reward functions  
13. **`step13_save_responsibilities.py`** — Export responsibilities  
14. **`step14_delay_analysis.py`** — Analyse delay posteriors  
15. **`step15_strategy_delay_plots.py`** — Visualise strategy refinements  
16. **`step16_time_delay_analysis.py`** — Temporal analysis of delays  
17. **`step17_behavioral_plots.py`** — Final behavioural visualisations

---

## ▶️ How to Run

### **Run full preprocessing → EM–MaxEnt → plots pipeline**
Note: Add appropiate options
```bash
python3 main.py
python3 main2.py
python3 main3.py
```

---

## Dataset

Dataset and intermediate csv files are to large for Github!
View here: https://drive.google.com/drive/folders/1Vpc12GK0RCBRU-KEDG5Op2Rm1o3xiB1W?usp=drive_link
