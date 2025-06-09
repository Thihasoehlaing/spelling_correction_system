# ✨ Spelling Correction System (NLP Assignment)

A GUI-based NLP application that detects and corrects **non-word**, **real-word**, and **grammar errors** using advanced natural language processing techniques and a domain-rich corpus.

## 🚀 Features

- ✅ **Non-word error detection** (e.g., `sientist` → `scientist`)
- ✅ **Real-word misuse detection** using **trigram context probability** (e.g., `sea` → `see`)
- ✅ **Grammar checking** (e.g., `He want` → `He wants`) using **spaCy POS tagging**
- ✅ **GUI interface** built with **PySide6**
- ✅ **Interactive word suggestions** via right-click context menu
- ✅ **Integrated dictionary view** with search and filter
- ✅ Supports **tense and subject-verb agreement checks** (present, past)

## 🧠 Techniques Used

| Error Type    | Technique(s) Used                                      |
|---------------|--------------------------------------------------------|
| Non-word      | `pyspellchecker`, NLTK `words`, edit distance          |
| Real-word     | NLTK `bigrams`, `trigrams`, context-aware probability  |
| Grammar       | spaCy POS tagging and dependency parsing (`nsubj`, `aux`) |

## 🖥️ GUI Preview

- ✍️ Input field with 500-character limit
- 🖱️ Right-click on underlined words to see suggestions
- 🔴 Red underline: Non-word error  
- 🔵 Blue underline: Real-word misuse  
- 🟢 Green underline: Grammar error  
- 📚 Right panel: Full searchable dictionary list


## 🧪 Sample Input

```text
He want to become a great sientist. Every day, he reserches new ideas about human behavior and medcine. 
Yesterday, he went to the lab to sea the results of his experiment. His findings were cleer and significant, 
but he forgot to write down the resluts.
```

## 🛠️ Installation
### 1. Clone the repository
```bash
git clone git@github.com:Thihasoehlaing/spelling-correction-system.git
cd spelling-correction-system
```

### 2. Install dependencies
Make sure you have Python 3.9–3.11. Python 3.12+ is not supported by some libraries.
```bash
pip install nltk spacy PySide6 pyspellchecker 
python -m nltk.downloader punkt brown words averaged_perceptron_tagger
python -m spacy download en_core_web_sm
```

### 3. Run the application
```bash
python main.py
```

## 📦 Requirements
- [NLTK](https://www.nltk.org/)
- [spaCy](https://spacy.io/)
- [pyspellchecker](https://pypi.org/project/pyspellchecker/)
- [PySide6](https://pypi.org/project/PySide6/)

## 📚 References
- NLTK Corpus & Language Models

- spaCy POS Tagging

- PySpellChecker for edit distance

- PySide6 for desktop GUI