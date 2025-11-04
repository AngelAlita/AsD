# Attack as Defense: Safeguarding Large Vision-Language Models from Jailbreaking by Adversarial Attacks
This is the official implementation of EMNLP 2025 paper [Attack as Defense: Safeguarding Large Vision-Language Models from Jailbreaking by Adversarial Attacks](https://aclanthology.org/2025.findings-emnlp.1095.pdf)


![overview](assests/overview.png)

## 📑 Abstract
Adversarial vulnerabilities in vision-language models pose a critical challenge to the reliability of large language systems, where typographic manipulations and adversarial perturbations can effectively bypass language model defenses.
We introduce Attack as Defense (AsD), the first approach to proactively defend at the cross-modality level, embedding protective perturbations in vision to disrupt attacks before they propagate to the language model. By leveraging the semantic alignment between vision and language, AsD enhances adversarial robustness through model perturbations and system-level prompting. Unlike prior work that focuses on text-stage defenses, our method integrates visual defenses to reinforce prompt-based protections, mitigating jailbreaking attacks across benchmarks. 
Experiments on the LLaVA-1.5 show that AsD reduces attack success rates from 56.7\% to 12.6\% for typographic attacks and from 89.0\% to 47.5\% for adversarial perturbations.
Further analysis reveals that the key bottleneck in vision-language security lies not in isolated model vulnerabilities, but in cross-modal interactions, where adversarial cues in the vision model fail to consistently activate the defense mechanisms of the language model.

## 📖 Datasets
We use the MM-SafetyBench dataset for Jailbreak Attack. You can download it from [here](https://github.com/isXinLiu/MM-SafetyBench) and put it in the data folder.


## ⚙️ Installation
```bash
git clone https://github.com/AngelAlita/AsD.git
cd AsD
conda create -n AsD python=3.10 -y
conda activate AsD

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
````

## 🔍 Usage
```bash
python AsD.py --data_type SD_TYPO --save_folder ./results/MMBench/AsD --AsD_img_path ./images/L2_noNorm_clipgrad_tips_336_LR0_1-1_1.png

python evaluation.py --data_type SD_TYPO --save_folder ./results/MMBench/AsD --AsD
```

## 🔖 Citation
If you find our repository helpful, kindly cite our paper: 
```bibtex
@inproceedings{li-etal-2025-attack,
    title = "Attack as Defense: Safeguarding Large Vision-Language Models from Jailbreaking by Adversarial Attacks",
    author = "Li, Chongxin  and
      Wang, Hanzhang  and
      Fang, Yuchun",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.1095/",
    pages = "20138--20152",
    ISBN = "979-8-89176-335-7"
}

```

## 🔑 License

This project is licensed under the [MIT License](LICENSE).

## 🙌 Acknowledgements

This work builds upon: [LLaVA](https://github.com/haotian-liu/LLaVA),[MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench) and[Jailbreak-in-piece](https://github.com/erfanshayegani/Jailbreak-In-Pieces). We thank the authors for their great work.



