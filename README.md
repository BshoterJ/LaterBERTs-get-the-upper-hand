# BERT系列魔改、搜索、剪枝、蒸馏方案

## 优化设计
### 预训练模型
- Deep Contextualized Word Representations (NAACL 2018) [[paper]](https://aclweb.org/anthology/N18-1202) - ***ELMo***
- Universal Language Model Fine-tuning for Text Classification (ACL 2018) [[paper]](https://www.aclweb.org/anthology/P18-1031/) - ***ULMFit***
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (NAACL 2019) [[paper]](https://www.aclweb.org/anthology/N19-1423)[[code]](https://github.com/google-research/bert)[[official PyTorch code]](https://github.com/codertimo/BERT-pytorch) - ***BERT***
- Improving Language Understanding by Generative Pre-Training (CoRR 2018) [[paper]](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) - ***GPT***
- Language Models are Unsupervised Multitask Learners (CoRR 2019) [[paper]](https://www.techbooky.com/wp-content/uploads/2019/02/Better-Language-Models-and-Their-Implications.pdf)[[code]](https://github.com/openai/gpt-2) - ***GPT-2***
- MASS: Masked Sequence to Sequence Pre-training for Language Generation (ICML 2019) [[paper]](http://proceedings.mlr.press/v97/song19d/song19d.pdf)[[code]](https://github.com/microsoft/MASS) - ***MASS***
- Unified Language Model Pre-training for Natural Language Understanding and Generation (CoRR 2019) [[paper]](https://arxiv.org/pdf/1905.03197.pdf)[[code]](https://github.com/microsoft/unilm) - ***UNILM*** 
- Multi-Task Deep Neural Networks for Natural Language Understanding (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-1441)[[code]](https://github.com/namisan/mt-dnn) - ***MT-DNN***
- 75 Languages, 1 Model: Parsing Universal Dependencies Universally[[paper]](https://www.aclweb.org/anthology/D19-1279/)[[code]](https://github.com/hyperparticle/udify) - ***UDify***
- ERNIE: Enhanced Language Representation with Informative Entities (ACL 2019) [[paper]](https://www.aclweb.org/anthology/P19-1139)[[code]](https://github.com/thunlp/ERNIE) - ***ERNIE (THU)***
- ERNIE: Enhanced Representation through Knowledge Integration (CoRR 2019) [[paper]](https://arxiv.org/pdf/1904.09223.pdf) - ***ERNIE (Baidu)***
- Defending Against Neural Fake News (CoRR 2019) [[paper]](https://arxiv.org/abs/1905.12616)[[code]](https://rowanzellers.com/grover/) - ***Grover***
- ERNIE 2.0: A Continual Pre-training Framework for Language Understanding (CoRR 2019) [[paper]](https://arxiv.org/pdf/1907.12412.pdf) - ***ERNIE 2.0 (Baidu)***
- Pre-Training with Whole Word Masking for Chinese BERT (CoRR 2019) [[paper]](https://arxiv.org/pdf/1906.08101.pdf) - ***Chinese-BERT-wwm***
- SpanBERT: Improving Pre-training by Representing and Predicting Spans (CoRR 2019) [[paper]](https://arxiv.org/pdf/1907.10529.pdf) - ***SpanBERT***
- XLNet: Generalized Autoregressive Pretraining for Language Understanding  (CoRR 2019) [[paper]](https://arxiv.org/pdf/1906.08237.pdf)[[code]](https://github.com/zihangdai/xlnet) - ***XLNet***
- RoBERTa: A Robustly Optimized BERT Pretraining Approach (CoRR 2019) [[paper]](https://arxiv.org/pdf/1907.11692.pdf) - ***RoBERTa***
- NEZHA: Neural Contextualized Representation for Chinese Language Understanding (CoRR 2019) [[paper]](https://arxiv.org/abs/1909.00204)[[code]](https://github.com/huawei-noah/Pretrained-Language-Model) - ***NEZHA***
- K-BERT: Enabling Language Representation with Knowledge Graph (AAAI 2020) [[paper]](https://arxiv.org/abs/1909.07606)[[code]](https://github.com/autoliuweijie/K-BERT) - ***K-BERT***
- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (CoRR 2019) [[paper]](https://arxiv.org/abs/1909.08053)[[code]](https://github.com/NVIDIA/Megatron-LM) - ***Megatron-LM***
- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transforme (CoRR 2019) [[paper]](https://arxiv.org/abs/1910.10683)[[code]](https://github.com/google-research/text-to-text-transfer-transformer) - ***T5***
- BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (CoRR 2019) [[paper]](https://arxiv.org/abs/1910.13461) - ***BART***
- ZEN: Pre-training Chinese Text Encoder Enhanced by N-gram Representations (CoRR 2019) [[paper]](https://arxiv.org/abs/1911.00720)[[code]](https://github.com/sinovation/zen) - ***ZEN***
- The JDDC Corpus: A Large-Scale Multi-Turn Chinese Dialogue Dataset for E-commerce Customer Service (CoRR 2019) [[paper]](https://arxiv.org/pdf/1911.09969.pdf)[[code]](https://github.com/jd-aig/nlp_baai) - ***BAAI-JDAI-BERT***
- Knowledge Enhanced Contextual Word Representations (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1005/) - ***KnowBert***
- UER: An Open-Source Toolkit for Pre-training Models (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-3041/)[[code]](https://github.com/dbiir/UER-py) - ***UER***
- ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators (ICLR 2020) [[paper]](https://openreview.net/forum?id=r1xMH1BtvB) - ***ELECTRA***
- StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding (ICLR 2020) [[paper]](https://arxiv.org/abs/1908.04577) - ***StructBERT***
- FreeLB: Enhanced Adversarial Training for Language Understanding (ICLR 2020) [[paper]](https://arxiv.org/abs/1909.11764)[[code]](https://github.com/zhuchen03/FreeLB) - ***FreeLB***
- HUBERT Untangles BERT to Improve Transfer across NLP Tasks (CoRR 2019) [[paper]](https://arxiv.org/pdf/1910.12647.pdf) - ***HUBERT***
- CodeBERT: A Pre-Trained Model for Programming and Natural Languages (CoRR 2020) [[paper]](https://arxiv.org/abs/2002.08155) - ***CodeBERT***
- ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training (CoRR 2020) [[paper]](https://arxiv.org/abs/2001.04063) - ***ProphetNet***
- ERNIE-GEN: An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation (CoRR 2020) [[paper]](https://arxiv.org/abs/2001.11314)[[code]](https://github.com/PaddlePaddle/ERNIE/tree/repro/ernie-gen) - ***ERNIE-GEN***
- Efficient Training of BERT by Progressively Stacking (ICML 2019) [[paper]](http://proceedings.mlr.press/v97/gong19a.html)[[code]](https://github.com/gonglinyuan/StackingBERT) - ***StackingBERT***
- PoWER-BERT: Accelerating BERT Inference via Progressive Word-vector Elimination (CoRR 2020) [[paper]](https://arxiv.org/abs/2001.08950)[[code]](https://github.com/IBM/PoWER-BERT)
- Towards a Human-like Open-Domain Chatbot (CoRR 2020) [[paper]](https://arxiv.org/abs/2001.09977) - ***Meena***
- UniLMv2: Pseudo-Masked Language Models for Unified Language Model Pre-Training (CoRR 2020) [[paper]](https://arxiv.org/abs/2002.12804)[[code]](https://github.com/microsoft/unilm) - ***UNILMv2***
- Optimus: Organizing Sentences via Pre-trained Modeling of a Latent Space (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.04092)[[code]](https://github.com/ChunyuanLI/Optimus) - ***Optimus***
- SegaBERT: Pre-training of Segment-aware BERT for Language Understanding. *He Bai, Peng Shi, Jimmy Lin, Luchen Tan, Kun Xiong, Wen Gao, Ming Li*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.14996)
- MPNet: Masked and Permuted Pre-training for Language Understanding (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.09297)[[code]](https://github.com/microsoft/MPNet) - ***MPNet***
- Language Models are Few-Shot Learners (CoRR 2020) [[paper]](https://arxiv.org/abs/2005.14165)[[code]](https://github.com/openai/gpt-3) - ***GPT-3***
- SPECTER: Document-level Representation Learning using Citation-informed Transformers (ACL 2020) [[paper]](https://arxiv.org/abs/2004.07180) - ***SPECTER***
- Recipes for building an open-domain chatbot (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.13637)[[post]](https://ai.facebook.com/blog/state-of-the-art-open-source-chatbot/)[[code]](https://github.com/facebookresearch/ParlAI) - ***Blender***
- PLATO-2: Towards Building an Open-Domain Chatbot via Curriculum Learning (CoRR 2020) [[paper]](https://arxiv.org/abs/2006.16779)[[code]](https://github.com/PaddlePaddle/Knover/tree/master/plato-2) - ***PLATO-2***
- DeBERTa: Decoding-enhanced BERT with Disentangled Attention (CoRR 2020) [[paper]](https://arxiv.org/abs/2006.03654)[[code]](https://github.com/microsoft/DeBERTa) - ***DeBERTa***
- **Big Bird: Transformers for Longer Sequences**. *Big Bird: Transformers for Longer Sequences*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2007.14062)

### 多模态
- VideoBERT: A Joint Model for Video and Language Representation Learning (ICCV 2019) [[paper]](https://arxiv.org/abs/1904.01766)
- Learning Video Representations using Contrastive Bidirectional Transformer (CoRR 2019) [[paper]](https://arxiv.org/abs/1906.05743) - ***CBT***
- ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks (NeurIPS 2019) [[paper]](https://arxiv.org/abs/1908.02265)[[code]](https://github.com/jiasenlu/vilbert_beta)
- VisualBERT: A Simple and Performant Baseline for Vision and Language (CoRR 2019) [[paper]](https://arxiv.org/abs/1908.03557)[[code]](https://github.com/uclanlp/visualbert)
- Fusion of Detected Objects in Text for Visual Question Answering (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1219/)[[code]](https://github.com/google-research/
language/tree/master/language/question_answering/b2t2) - ***B2T2***
- Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training (AAAI 2020) [[paper]](https://arxiv.org/abs/1908.06066)
- LXMERT: Learning Cross-Modality Encoder Representations from Transformers (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1514/)[[code]](https://github.com/airsplay/lxmert)
- VL-BERT: Pre-training of Generic Visual-Linguistic Representatio (CoRR 2019) [[paper]](https://arxiv.org/abs/1908.08530)[[code]](https://github.com/jackroos/VL-BERT)
- UNITER: Learning UNiversal Image-TExt Representations (CoRR 2019) [[paper]](https://arxiv.org/abs/1909.11740)
- FashionBERT: Text and Image Matching with Adaptive Loss for Cross-modal Retrieval （SIGIR 2020) [[paper]](https://arxiv.org/abs/2005.09801) - ***FashionBERT***
- VD-BERT: A Unified Vision and Dialog Transformer with BERT (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.13278) - ***VD-BERT***


## 模型压缩
- Distilling Task-Specific Knowledge from BERT into Simple Neural Networks. *Raphael Tang, Yao Lu, Linqing Liu, Lili Mou, Olga Vechtomova, Jimmy Lin*. (CoRR 2019) [[paper]](https://arxiv.org/abs/1903.12136)
- Model Compression with Multi-Task Knowledge Distillation for Web-scale Question Answering System. *Ze Yang, Linjun Shou, Ming Gong, Wutao Lin, Daxin Jiang*. (CoRR 2019) [[paper]](https://arxiv.org/abs/1904.09636) - ***MKDM***
- Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding. *Xiaodong Liu, Pengcheng He, Weizhu Chen, Jianfeng Gao*. (CoRR 2019) [[paper]](https://arxiv.org/abs/1904.09482)
- Well-Read Students Learn Better: On the Importance of Pre-training Compact Models. *Iulia Turc, Ming-Wei Chang, Kenton Lee, Kristina Toutanova*. (CoRR 2019) [[paper]](https://arxiv.org/abs/1908.08962)
- Small and Practical BERT Models for Sequence Labeling. *Henry Tsai, Jason Riesa, Melvin Johnson, Naveen Arivazhagan, Xin Li, Amelia Archer*. (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1374/)
- Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT. *Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei Yao, Amir Gholami, Michael W. Mahoney, Kurt Keutzer*. (AAAI 2020) [[paper]](https://arxiv.org/abs/1909.05840)
- Patient Knowledge Distillation for BERT Model Compression. *Siqi Sun, Yu Cheng, Zhe Gan, Jingjing Liu*. (EMNLP 2019) [[paper]](https://www.aclweb.org/anthology/D19-1441/) - ***BERT-PKD***
- Extreme Language Model Compression with Optimal Subwords and Shared Projections. *Sanqiang Zhao, Raghav Gupta, Yang Song, Denny Zhou*. (ICLR 2019) [[paper]](https://arxiv.org/abs/1909.11687)
- DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf*. [[paper]](https://arxiv.org/pdf/1910.01108.pdf)[[code]](https://github.com/huggingface/transformers/tree/master/examples/distillation)
- TinyBERT: Distilling BERT for Natural Language Understanding. *Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu*. (ICLR 2019) [[paper]](https://arxiv.org/pdf/1909.10351.pdf)[[code]](https://github.com/huawei-noah/Pretrained-Language-Model)
- Q8BERT: Quantized 8Bit BERT. *Ofir Zafrir, Guy Boudoukh, Peter Izsak, Moshe Wasserblat*. (NeurIPS 2019 Workshop) [[paper]](https://arxiv.org/abs/1910.06188)
- ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. *Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut*. (ICLR 2020) [[paper]](https://arxiv.org/abs/1909.11942)[[code]](https://github.com/google-research/ALBERT)
- Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning. *Mitchell A. Gordon, Kevin Duh, Nicholas Andrews*. (ICLR 2020) [[paper]](https://openreview.net/forum?id=SJlPOCEKvH)[[PyTorch code]](https://github.com/lonePatient/albert_pytorch)
- Reducing Transformer Depth on Demand with Structured Dropout. *Angela Fan, Edouard Grave, Armand Joulin*. (ICLR 2020) [[paper]](https://arxiv.org/abs/1909.11556) - ***LayerDrop***
- Multilingual Alignment of Contextual Word Representations (ICLR 2020) [[paper]](https://arxiv.org/abs/2002.03518)
- AdaBERT: Task-Adaptive BERT Compression with Differentiable Neural Architecture Search. *Daoyuan Chen, Yaliang Li, Minghui Qiu, Zhen Wang, Bofang Li, Bolin Ding, Hongbo Deng, Jun Huang, Wei Lin, Jingren Zhou*. (IJCAI 2020) [[paper]](https://arxiv.org/pdf/2001.04246.pdf) - ***AdaBERT***
- BERT-of-Theseus: Compressing BERT by Progressive Module Replacing. *Canwen Xu, Wangchunshu Zhou, Tao Ge, Furu Wei, Ming Zhou*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2002.02925)[[pt code]](https://github.com/JetRunner/BERT-of-Theseus)[[tf code]](https://github.com/qiufengyuyi/bert-of-theseus-tf)[[keras code]](https://github.com/bojone/bert-of-theseus)
- MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2002.10957)[[code]](https://github.com/microsoft/unilm/tree/master/minilm)
- FastBERT: a Self-distilling BERT with Adaptive Inference Time. *Weijie Liu, Peng Zhou, Zhiruo Wang, Zhe Zhao, Haotang Deng, Qi Ju*. (ACL 2020) [[paper]](https://arxiv.org/abs/2004.02178)[[code]](https://github.com/autoliuweijie/FastBERT)
- MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices. *Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, Denny Zhou*. (ACL 2020) [[paper]](https://arxiv.org/abs/2004.02984)[[code]](https://github.com/google-research/google-research/tree/master/mobilebert)
- Towards Non-task-specific Distillation of BERT via Sentence Representation Approximation. *Bowen Wu, Huan Zhang, Mengyuan Li, Zongsheng Wang, Qihang Feng, Junhong Huang, Baoxun Wang*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.03097) - ***BiLSTM-SRA & LTD-BERT***
- Poor Man's BERT: Smaller and Faster Transformer Models. *Hassan Sajjad, Fahim Dalvi, Nadir Durrani, Preslav Nakov*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.03844)
- DynaBERT: Dynamic BERT with Adaptive Width and Depth. *Lu Hou, Lifeng Shang, Xin Jiang, Qun Liu*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2004.04037)
- SqueezeBERT: What can computer vision teach NLP about efficient neural networks?. *Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, Kurt W. Keutzer*. (CoRR 2020) [[paper]](https://arxiv.org/abs/2006.11316)

## 模型搜索
