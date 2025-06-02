import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QTextEdit, QPushButton,
    QLabel, QListWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor

import nltk
from nltk.corpus import words as nltk_words, brown
from nltk.tokenize import word_tokenize
from nltk import bigrams, trigrams, pos_tag
from spellchecker import SpellChecker
from collections import Counter

# Download required NLTK data
nltk.download('punkt')
nltk.download('words')
nltk.download('brown')
nltk.download('averaged_perceptron_tagger')

# === Spell Checker Setup ===
spell = SpellChecker()
dictionary = set(word.lower() for word in nltk_words.words() if word.isalpha())

# === Bigram/Trigram Language Model ===
brown_words = [word.lower() for word in brown.words() if word.isalpha()]
bigram_counts = Counter(bigrams(brown_words))
trigram_counts = Counter(trigrams(brown_words))
unigram_counts = Counter(brown_words)

def bigram_prob(w1, w2):
    return bigram_counts[(w1, w2)] / unigram_counts[w1] if unigram_counts[w1] > 0 else 0

def trigram_prob(w1, w2, w3):
    return trigram_counts[(w1, w2, w3)] / bigram_counts[(w1, w2)] if bigram_counts[(w1, w2)] > 0 else 0

def check_word(word):
    if word not in dictionary:
        suggestions = spell.candidates(word)
        return False, list(suggestions)[:3], "non-word"
    return True, [], "valid"

def check_sentence(text):
    tokens = [w.lower() for w in word_tokenize(text)]
    tags = pos_tag(tokens)
    errors = {}

    for i, (word, tag) in enumerate(tags):
        if word.isalpha():
            if word not in dictionary:
                correct, suggestions, error_type = check_word(word)
                if not correct:
                    errors[word] = (suggestions, error_type)
            elif i > 1:
                w1, w2 = tokens[i - 2], tokens[i - 1]
                actual_prob = trigram_prob(w1, w2, word)
                candidates = spell.known(spell.edit_distance_1(word))
                ranked = sorted(
                    [w for w in candidates if w in dictionary and w != word],
                    key=lambda w: trigram_prob(w1, w2, w),
                    reverse=True
                )
                if ranked and trigram_prob(w1, w2, ranked[0]) > actual_prob * 5:
                    errors[word] = ([ranked[0]], "real-word")
    return errors

# === GUI Application ===
class SpellCheckerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spelling Correction System")
        self.setFixedSize(800, 500)
        self.error_positions = []

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Write something here (max 500 characters)...")
        self.text_edit.textChanged.connect(self.limit_text)

        self.check_button = QPushButton("Check Spelling")
        self.check_button.clicked.connect(self.check_spelling)

        self.result_label = QLabel("Spelling Suggestions:")
        self.result_list = QListWidget()

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.check_button)
        left_layout.addWidget(self.text_edit)
        left_layout.addWidget(self.result_label)
        left_layout.addWidget(self.result_list)

        self.dict_label = QLabel("Dictionary:")
        self.dictionary_list = QListWidget()
        self.dictionary_words = sorted(dictionary)
        self.dictionary_list.addItems(self.dictionary_words)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search word...")
        self.search_input.textChanged.connect(self.search_dictionary)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.dict_label)
        right_layout.addWidget(self.dictionary_list)
        right_layout.addWidget(QLabel("Search"))
        right_layout.addWidget(self.search_input)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)
        self.setLayout(main_layout)

    def limit_text(self):
        text = self.text_edit.toPlainText()
        if len(text) > 500:
            self.text_edit.setPlainText(text[:500])
            cursor = self.text_edit.textCursor()
            cursor.setPosition(500)
            self.text_edit.setTextCursor(cursor)

    def check_spelling(self):
        self.error_positions.clear()
        text = self.text_edit.toPlainText()
        self.result_list.clear()

        cursor = self.text_edit.textCursor()
        cursor.select(QTextCursor.Document)
        cursor.setCharFormat(QTextCharFormat())

        if not text.strip():
            QMessageBox.warning(self, "Empty Text", "Please write something first.")
            return

        errors = check_sentence(text)

        if not errors:
            self.result_list.addItem("No spelling errors found.")
        else:
            for word, (suggestions, error_type) in errors.items():
                self.highlight_and_track_word(word, error_type)
                self.result_list.addItem(f"{word} ➜ Suggestions: {', '.join(suggestions)}")

    def highlight_and_track_word(self, word, error_type):
        format = QTextCharFormat()
        if error_type == "non-word":
            format.setUnderlineColor(QColor("red"))
        elif error_type == "real-word":
            format.setUnderlineColor(QColor("orange"))
        format.setUnderlineStyle(QTextCharFormat.SpellCheckUnderline)

        text = self.text_edit.toPlainText()
        cursor = self.text_edit.textCursor()
        position = 0
        while True:
            position = text.lower().find(word.lower(), position)
            if position == -1:
                break
            cursor.setPosition(position)
            cursor.setPosition(position + len(word), QTextCursor.KeepAnchor)
            cursor.setCharFormat(format)
            self.error_positions.append((word, position, position + len(word)))
            position += len(word)

    def search_dictionary(self):
        query = self.search_input.text().lower()
        self.dictionary_list.clear()
        filtered = [word for word in self.dictionary_words if query in word]
        self.dictionary_list.addItems(filtered)

# Run the app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpellCheckerApp()
    window.show()
    sys.exit(app.exec())
