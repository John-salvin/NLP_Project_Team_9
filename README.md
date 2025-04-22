# NLP_Project_Team_9
Sentiment analysis for Telugu text - classifying text as positive, negative or neutral.

# Telugu Sentiment Analysis

A comprehensive toolkit for analyzing sentiment in Telugu text, featuring traditional machine learning and deep learning approaches with extensive feature engineering.

## Overview

This project provides a complete pipeline for Telugu sentiment analysis, from data preprocessing to advanced modeling. It includes specialized techniques for Telugu text normalization, tokenization, and feature extraction, as well as implementations of both traditional machine learning models and neural networks for sentiment classification.

## Features

- Complete Telugu text preprocessing pipeline
- Extensive feature engineering including:
  - Linguistic features (POS tagging, word statistics)
  - Semantic features (word embeddings, sentiment lexicons)
  - Clustering-based features
- Multiple model implementations:
  - Traditional ML (Random Forest, Logistic Regression)
  - Neural Networks (Hybrid models, LSTM, Bidirectional LSTM)
- Comprehensive evaluation metrics and visualizations

## Configuration and Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for neural network training)

### Installation

#### Google Colab (Recommended)

1. Upload the `Nlp_project_Final.ipynb` notebook to Google Colab
2. The notebook is designed to run in Google Colab with GPU acceleration
3. Required packages will be installed automatically through the notebook

#### Local System Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Cha14ran/DREAM-T.git
   cd DREAM-T
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install indic-nlp-library advertools gensim stanza fasttext torch pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn wordcloud
   ```

4. Download external resources:
   ```bash
   # Clone Indic NLP resources
   git clone https://github.com/anoopkunchukuttan/indic_nlp_resources
   
   # Download FastText Telugu embeddings
   wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.te.300.bin.gz
   gunzip cc.te.300.bin.gz
   
   # Download Stanza Telugu model
   python -c "import stanza; stanza.download('te')"
   ```

### Required Packages

The following packages are required:

```
pandas>=1.1.0
numpy>=1.23.5
indic-nlp-library>=0.92
advertools>=0.16.6
gensim>=4.3.0
stanza>=1.10.0
fasttext>=0.9.2
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
torch>=1.10.0
xgboost>=1.5.0
imbalanced-learn>=0.8.0
wordcloud>=1.8.0
```

## File Manifest

```
telugu-sentiment-analysis/
├── Nlp_project_Final.ipynb     # Main notebook with complete pipeline
├── data/
│   ├── pos.txt                 # Positive sentiment words in Telugu
│   ├── neg.txt                 # Negative sentiment words in Telugu
│   ├── pos_tagging.csv         # Intermediate data with POS tags
│   ├── final_df.csv            # Processed data with all features
│   └── processed_train_df.csv  # Normalized features for model training
├── indic_nlp_resources/        # Resources for Telugu NLP processing
├── cc.te.300.bin               # FastText Telugu word embeddings
├── NotoSansTelugu-Regular.ttf  # Telugu font for visualizations
└── README.md
```

## Required External Files

The following files need to be downloaded separately:

1. **Telugu Sentiment Dataset**:
   - Source: [mounikaiiith/Telugu_Sentiment](https://huggingface.co/datasets/mounikaiiith/Telugu_Sentiment/viewer)
   - Files: `Sentiment_train.csv`, `Sentiment_valid.csv`, `Sentiment_test.csv`
   - The notebook automatically downloads these from Hugging Face

2. **FastText Telugu Embeddings**:
   - File: `cc.te.300.bin`
   - Source: [Facebook FastText](https://fasttext.cc/docs/en/crawl-vectors.html)
   - Download Command: 
     ```
     wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.te.300.bin.gz
     gunzip cc.te.300.bin.gz
     ```

3. **Positive and Negative Word Lists**:
   - Files: `pos.txt`, `neg.txt`
   - Source: [DREAM-T Repository](https://github.com/Cha14ran/DREAM-T/tree/master)
   - These files contain Telugu words with positive and negative sentiment

4. **Indic NLP Resources**:
   - Repository: [indic_nlp_resources](https://github.com/anoopkunchukuttan/indic_nlp_resources)
   - Clone Command:
     ```
     git clone https://github.com/anoopkunchukuttan/indic_nlp_resources
     ```

5. **Telugu Font** (for visualizations):
   - File: `NotoSansTelugu-Regular.ttf`
   - Source: [Google Noto Fonts](https://fonts.google.com/noto/specimen/Noto+Sans+Telugu)

## Operating Instructions

### Running in Google Colab (Recommended)

1. Upload the `Nlp_project_Final.ipynb` notebook to Google Colab
2. Upload the required external files (`pos.txt`, `neg.txt`) to your Colab session
3. Run the cells in order, as the notebook will:
   - Install all required dependencies
   - Download the Telugu Sentiment dataset
   - Clone the Indic NLP resources
   - Download the FastText model
   - Process the data and train models

### Running Locally

1. Ensure all prerequisites and external files are in place
2. Set up the correct paths in the notebook:
   ```python
   # Update these paths for local execution
   os.environ['INDIC_RESOURCES_PATH'] = '/path/to/indic_nlp_resources'
   ```
3. Run the notebook cells in order

## Pipeline Workflow

The notebook implements the following workflow:

1. **Data Loading**: Load Telugu sentiment dataset from Hugging Face
2. **Text Preprocessing**:
   - Telugu script normalization
   - Tokenization
   - Stopword removal
3. **Feature Engineering**:
   - Basic statistical features (sentence length, word length)
   - Word embeddings using Word2Vec
   - POS tagging features
   - Sentiment lexicon features
   - Clustering features
4. **Model Training**:
   - Traditional ML models (Logistic Regression, Random Forest)
   - Neural network models (Hybrid NN, LSTM, Bidirectional LSTM)
5. **Model Evaluation**:
   - Classification metrics
   - Feature importance analysis

## Dataset Information

This project uses the Telugu Sentiment dataset available on Hugging Face:

- **Dataset**: [mounikaiiith/Telugu_Sentiment](https://huggingface.co/datasets/mounikaiiith/Telugu_Sentiment/viewer)
- **Description**: A collection of Telugu sentences labeled with sentiment (positive, negative, or neutral)
- **Files**:
  - `Sentiment_train.csv`: Training data
  - `Sentiment_valid.csv`: Validation data
  - `Sentiment_test.csv`: Test data

## Known Issues and Limitations

1. **Class Imbalance**: The dataset has an imbalanced distribution of sentiment classes, with neutral samples being overrepresented. This may affect model performance.

2. **Processing Time**: Feature extraction, particularly POS tagging and sentiment lexicon matching, can be time-consuming for large datasets.

3. **Memory Usage**: Loading and processing word embeddings requires significant memory, especially for the FastText model.

4. **Telugu Script Variations**: Some Telugu script variations may not be properly normalized, potentially affecting tokenization quality.

5. **Numpy Version Compatibility**: The notebook requires numpy==1.23.5 for compatibility with gensim. This may cause conflicts with other packages that require newer numpy versions.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```
   ModuleNotFoundError: No module named 'indic_nlp_library'
   ```
   **Solution**: Ensure all dependencies are installed using the installation commands provided.

2. **CUDA Out of Memory**:
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce batch size in the neural network training configuration or use a smaller model.

3. **Indic NLP Resources Path Error**:
   ```
   FileNotFoundError: [Errno 2] No such file or directory: '/content/indic_nlp_resources'
   ```
   **Solution**: Set the correct path to the Indic NLP resources directory:
   ```python
   import os
   os.environ['INDIC_RESOURCES_PATH'] = '/path/to/indic_nlp_resources'
   ```

4. **FastText Model Loading Error**:
   ```
   ValueError: fasttext_model: cannot open file
   ```
   **Solution**: Ensure the FastText model file (`cc.te.300.bin`) is in the correct location and has the correct permissions.

5. **Numpy Version Conflicts**:
   ```
   ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
   ```
   **Solution**: In Colab, this warning can be ignored. For local installation, consider using a dedicated virtual environment.

### Performance Optimization

1. **Slow POS Tagging**: 
   - Use batch processing for POS tagging
   - Cache POS tags for frequently used words

2. **Memory Issues with Word Embeddings**:
   - Use dimensionality reduction techniques like PCA
   - Load only required word vectors instead of the entire model

3. **Slow Neural Network Training**:
   - Use mixed precision training
   - Implement early stopping
   - Use a smaller subset of features

## Copyright and Licensing

This project is based on the [DREAM-T repository](https://github.com/Cha14ran/DREAM-T) and follows its licensing terms.

The Telugu Sentiment dataset is provided under its original license. Please refer to the [dataset page](https://huggingface.co/datasets/mounikaiiith/Telugu_Sentiment) for more information.

## Contact Information

For questions or issues related to this project, please contact the original repository maintainer:
- GitHub: [https://github.com/Cha14ran](https://github.com/Cha14ran)

## Credits and Acknowledgments

- **Telugu Sentiment Dataset**: [Mounika IIITH](https://huggingface.co/datasets/mounikaiiith/Telugu_Sentiment)
- **Indic NLP Library**: [Anoop Kunchukuttan](https://github.com/anoopkunchukuttan/indic_nlp_library)
- **FastText**: [Facebook Research](https://fasttext.cc/)
- **Stanza**: [Stanford NLP Group](https://stanfordnlp.github.io/stanza/)
- **DREAM-T Repository**: [Cha14ran](https://github.com/Cha14ran/DREAM-T)

Special thanks to the open-source community for providing tools and resources for Telugu NLP research.

## References

1. B. Sunitha and K. Madhavi. "Telugu Sentiwordnet: A lexical resource for sentiment analysis in Telugu." In 2018 Second International Conference on Computing Methodologies and Communication (ICCMC), pp. 659-663. IEEE, 2018.

2. R. S. Reddy and A. J. Reddy. "A hybrid approach for Telugu sentiment analysis." In Proceedings of the International Conference on Intelligent Computing and Control Systems (ICICCS), pp. 593-598. IEEE, 2018.

3. Kunchukuttan, A., Mehta, P., & Bhattacharyya, P. (2018). The IIT Bombay English-Hindi Parallel Corpus. In Proceedings of LREC 2018.

4. Cha14ran. (2023). DREAM-T: Deep Learning Resources for the Evaluation and Analysis of Multilingual Text. GitHub repository. https://github.com/Cha14ran/DREAM-T

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/30779377/62013d89-8084-49d2-b723-4c9c0fee149a/paste.txt

---
Answer from Perplexity: pplx.ai/share
