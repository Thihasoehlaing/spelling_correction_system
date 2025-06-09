import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QTextEdit, QPushButton, QLabel,
    QListWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox, QMenu
)
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor, QAction
from PySide6.QtCore import Qt

import nltk
from nltk.corpus import brown, words
from nltk import word_tokenize, bigrams, trigrams
from spellchecker import SpellChecker
from collections import Counter

import spacy

# NLTK setup
nltk.download('punkt')
nltk.download('brown')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

# Load NLP models
nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()

# Corpus and language models
corpus_words = [w.lower() for w in brown.words() if w.isalpha()]
dictionary = set(words.words()) | set(corpus_words)
bigrams_c = Counter(bigrams(corpus_words))
trigrams_c = Counter(trigrams(corpus_words))

def trigram_prob(w1, w2, w3):
    return trigrams_c[(w1, w2, w3)] / bigrams_c[(w1, w2)] if bigrams_c[(w1, w2)] > 0 else 0

def check_text_advanced(text):
    doc = nlp(text)
    tokens = [t.text for t in doc]
    lower_tokens = [t.lower() for t in tokens]
    errors = {}

    for i, token in enumerate(doc):
        word = token.text
        if not word.isalpha():
            continue

        lw = word.lower()

        # 1. Non-word
        if lw not in dictionary:
            suggestions = list(spell.candidates(lw))[:3]
            errors[lw] = (suggestions, "non-word")

        # 2. Real-word misuse (trigram)
        elif i >= 2:
            w1, w2 = lower_tokens[i - 2], lower_tokens[i - 1]
            actual_score = trigram_prob(w1, w2, lw)
            candidates = spell.known(spell.edit_distance_1(lw))
            ranked = sorted(
                [w for w in candidates if w != lw],
                key=lambda w: trigram_prob(w1, w2, w),
                reverse=True
            )
            if ranked and trigram_prob(w1, w2, ranked[0]) > actual_score * 5:
                errors[lw] = ([ranked[0]], "real-word")

        # 3. Grammar & tense
        if token.tag_ in ["VBP", "VBZ", "VBD", "VB"] and token.head.pos_ in ["NOUN", "PRON"]:
            expected = token._.inflect(token.tag_)
            if expected and expected.lower() != word.lower():
                errors[lw] = ([expected], "grammar")

    return errors

# === GUI ===
class CustomTextEdit(QTextEdit):
    def __init__(self):
        super().__init__()
        self.suggestions_map = {}

    def set_suggestions_map(self, data):
        self.suggestions_map = data

    def contextMenuEvent(self, event):
        cursor = self.cursorForPosition(event.pos())
        cursor.select(QTextCursor.WordUnderCursor)
        word = cursor.selectedText().lower()

        if word in self.suggestions_map:
            menu = QMenu(self)
            for suggestion in self.suggestions_map[word][0]:
                action = QAction(suggestion, self)
                action.triggered.connect(lambda _, s=suggestion, c=cursor: self.replace_word(c, s))
                menu.addAction(action)
            menu.exec(event.globalPos())
        else:
            super().contextMenuEvent(event)

    def replace_word(self, cursor, suggestion):
        cursor.beginEditBlock()
        cursor.removeSelectedText()
        cursor.insertText(suggestion)
        cursor.endEditBlock()

class SpellCheckerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spelling Correction System")
        self.setFixedSize(900, 600)

        self.text_edit = CustomTextEdit()
        self.text_edit.setPlaceholderText("Write up to 500 characters...")
        self.text_edit.textChanged.connect(self.limit_text)

        self.check_button = QPushButton("Check Text")
        self.check_button.clicked.connect(self.check_text)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_text)

        self.result_list = QListWidget()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search dictionary...")
        self.search_input.textChanged.connect(self.search_dictionary)

        self.dictionary_list = QListWidget()
        self.dictionary_words = sorted(dictionary)
        self.dictionary_list.addItems(self.dictionary_words)

        # Layouts
        left = QVBoxLayout()
        left.addWidget(self.check_button)
        left.addWidget(self.clear_button)
        left.addWidget(self.text_edit)
        left.addWidget(QLabel("Suggestions"))
        left.addWidget(self.result_list)

        right = QVBoxLayout()
        right.addWidget(QLabel("Dictionary"))
        right.addWidget(self.search_input)
        right.addWidget(self.dictionary_list)

        main = QHBoxLayout()
        main.addLayout(left, 2)
        main.addLayout(right, 1)
        self.setLayout(main)

    def limit_text(self):
        text = self.text_edit.toPlainText()
        if len(text) > 500:
            self.text_edit.setPlainText(text[:500])
            cursor = self.text_edit.textCursor()
            cursor.setPosition(500)
            self.text_edit.setTextCursor(cursor)

    def clear_text(self):
        self.text_edit.clear()
        self.result_list.clear()

    def check_text(self):
        self.result_list.clear()
        text = self.text_edit.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "Empty", "Please enter some text.")
            return

        errors = check_text_advanced(text)
        self.text_edit.set_suggestions_map(errors)
        self.highlight_errors(errors)

        if not errors:
            self.result_list.addItem("No errors found.")
        else:
            for word, (sugs, kind) in errors.items():
                self.result_list.addItem(f"{word} ({kind}) ➜ {', '.join(sugs)}")

    def highlight_errors(self, errors):
        cursor = self.text_edit.textCursor()
        cursor.select(QTextCursor.Document)
        cursor.setCharFormat(QTextCharFormat())  # clear

        text = self.text_edit.toPlainText().lower()
        for word, (_, kind) in errors.items():
            fmt = QTextCharFormat()
            fmt.setUnderlineStyle(QTextCharFormat.SpellCheckUnderline)
            if kind == "non-word":
                fmt.setUnderlineColor(QColor("red"))
            elif kind == "real-word":
                fmt.setUnderlineColor(QColor("blue"))
            elif kind == "grammar":
                fmt.setUnderlineColor(QColor("green"))

            start = 0
            while True:
                start = text.find(word, start)
                if start == -1:
                    break
                cursor.setPosition(start)
                cursor.setPosition(start + len(word), QTextCursor.KeepAnchor)
                cursor.setCharFormat(fmt)
                start += len(word)

    def search_dictionary(self):
        query = self.search_input.text().lower()
        filtered = [w for w in self.dictionary_words if query in w]
        self.dictionary_list.clear()
        self.dictionary_list.addItems(filtered)

# Run
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpellCheckerApp()
    window.show()
    sys.exit(app.exec())
