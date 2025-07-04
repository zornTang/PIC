# PIC: Protein Importance Calculator

## Abstract
Human essential proteins (HEPs) are indispensable for individual viability and development. However, experimental methods to identify HEPs are often costly, time-consuming and labor-intensive. In addition, existing computational methods predict HEPs only at the cell line level, but HEPs vary across living human, cell lines and animal models. To address this, we develop a sequence-based deep learning model, PIC, by fine-tuning a pre-trained protein language model. PIC not only significantly outperforms existing methods for predicting HEPs but also provides comprehensive prediction results across three levels: human, mouse and cell line. Further, we define the protein essential score (PES), derived from PIC, to quantify human protein essentiality, and validate its effectiveness by a series of biological analyses. We demonstrate the biomedical value of PES by identifying novel potential prognostic biomarkers for breast cancer and quantifying the essentiality of 617462 human microproteins. 
![Overview](Workflow.png)


## Web server
PIC web server is now available at http://www.cuilab.cn/pic


## Publication
[Comprehensive prediction and analysis of human protein essentiality based on a pre-trained protein large language model](https://www.biorxiv.org/content/10.1101/2024.03.26.586900v1)
## Main requirements
* python=3.10.14
* pytorch=1.12.1
* torchaudio=0.12.1
* torchvision=0.13.1
* cudatoolkit=11.3.1
* scikit-learn=1.3.2
* pandas=2.1.1
* numpy=1.26.0
* fair-esm=2.0.0
## Usage
A demo for training and using PIC models using linux-64 platform

**Step1: clone the repo**
```
git clone https://github.com/KangBoming/PIC.git
cd PIC
```

**Step2: create and activate the environment**
```
cd PIC
conda env create -f environment.yml
conda activate PIC
unset LD_LIBRARY_PATH
```

**Step3: download pretrained protein language model**
```
cd pretrained_model
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
wget https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt
```

**Step4: extract the sequence embedding from raw protein sequences** 

The extracted sequence embeddng will be saved at file folder './result/seq_embedding'

* human-level
```
python ./code/embedding.py --data_path ./data/human_data.pkl --fasta_file ./result/protein_sequence.fasta --model ./pretrained_model/esm2_t33_650M_UR50D.pt --label_name human --output_dir ./result/seq_embedding --device cuda:7 --truncation_seq_length 1024
```

* mouse-level
```
python ./code/embedding.py --data_path ./data/mouse_data.pkl --fasta_file ./result/protein_sequence.fasta --model ./pretrained_model/esm2_t33_650M_UR50D.pt --label_name mouse --output_dir ./result/seq_embedding --device cuda:7 --truncation_seq_length 1024
```

* cell-level
```
python ./code/embedding.py --data_path ./data/cell_data.pkl --fasta_file ./result/protein_sequence.fasta --model ./pretrained_model/esm2_t33_650M_UR50D.pt --label_name A549 --output_dir ./result/seq_embedding --device cuda:7 --truncation_seq_length 1024
```

**Step5: train model**

The trained model will be saved at file folder './result/model_train_results'

* human-level
```
python ./code/main.py --data_path ./data/human_data.pkl --feature_dir ./result/seq_embedding --label_name human --save_path ./result/model_train_results 
```

* mouse-level
```
python ./code/main.py --data_path ./data/mouse_data.pkl --feature_dir ./result/seq_embedding --label_name mouse --save_path ./result/model_train_results 
```

* cell-level (single cell line)
```
python ./code/main.py --data_path ./data/cell_data.pkl --feature_dir ./result/seq_embedding --label_name A549 --save_path ./result/model_train_results 
```

* cell-level (multiple cell lines)
```
python ./code/train_all_cell_lines.py --specific_cell_lines "A549,HeLa,MCF7" --device cuda:7
```

**Step6: predict protein essentiality**

* single model prediction
```
python ./code/predict.py --model_path ./result/model_train_results/PIC_A549/PIC_A549_model.pth --input_fasta proteins.fasta --output_file results.csv
```

* ensemble model prediction (multiple cell lines)
```
python ./code/predict.py --ensemble_mode --model_dir ./result/model_train_results --input_fasta proteins.fasta --output_file ensemble_results.csv
```

Tips: You can set the `label_name` parameter to the name of any cell line (you can obtain the name of each cell line from the `data/cell_line_meta_info.csv` file) to train the corresponding cell-level PIC model. For prediction, you can use either single models or ensemble multiple models for better performance. 



## License
This project is licensed under the MIT License - see the [LICENSE.txt](https://github.com/KangBoming/PIC/blob/main/LICENSE) file for details


## Contact
Please feel free to contact us for any further queations

Boming Kang <kangbm@bjmu.edu.cn>

Qinghua Cui <cuiqinghua@bjmu.edu.cn>


