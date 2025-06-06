import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QTextEdit, QPushButton, QLabel,
    QListWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox, QMenu
)
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor, QAction
from PySide6.QtCore import Qt

import nltk
from nltk.corpus import brown, words
from nltk import word_tokenize, bigrams, trigrams, pos_tag
from spellchecker import SpellChecker
from collections import Counter

# Download required NLTK data
nltk.download('punkt')
nltk.download('brown')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

# NLP Model Setup
corpus_words = [w.lower() for w in brown.words() if w.isalpha()]
unigrams = Counter(corpus_words)
bigrams_c = Counter(bigrams(corpus_words))
trigrams_c = Counter(trigrams(corpus_words))

dictionary = set(w.lower() for w in words.words() if w.isalpha())
dictionary.update(w.lower() for w in brown.words() if w.isalpha())

spell = SpellChecker()

def trigram_prob(w1, w2, w3):
    return trigrams_c[(w1, w2, w3)] / bigrams_c[(w1, w2)] if bigrams_c[(w1, w2)] > 0 else 0

def check_text_nlp(text):
    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)
    errors = {}

    for i, (word, tag) in enumerate(tagged):
        if not word.isalpha():
            continue

        # Non-word error
        if word not in dictionary:
            suggestions = spell.candidates(word)
            errors[word] = (list(suggestions)[:3], "non-word")

        # Real-word misuse detection
        elif i >= 2:
            w1, w2 = tokens[i - 2], tokens[i - 1]
            actual_score = trigram_prob(w1, w2, word)

            candidates = spell.known(spell.edit_distance_1(word))
            ranked = sorted(
                [w for w in candidates if w in dictionary and w != word and unigrams[w] > 3],
                key=lambda w: trigram_prob(w1, w2, w),
                reverse=True
            )

            if ranked:
                best = ranked[0]
                best_score = trigram_prob(w1, w2, best)
                if actual_score < 0.00005 and best_score > actual_score * 8 and unigrams[word] < 300:
                    errors[word] = ([best], "real-word")

        # Grammar check
        if i >= 2:
            prev_word, prev_tag = tagged[i - 1]
            prev_prev_word, prev_prev_tag = tagged[i - 2]
            if prev_prev_tag == 'PRP' and prev_tag == 'TO' and tag in ['VB', 'VBP'] and word.endswith('s') is False:
                errors[word] = ([word + 's'], "grammar")

    return errors

class CustomTextEdit(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.suggestions_map = {}

    def set_suggestions_map(self, data):
        self.suggestions_map = data

    def contextMenuEvent(self, event):
        cursor = self.cursorForPosition(event.pos())
        cursor.select(QTextCursor.WordUnderCursor)
        word = cursor.selectedText().lower()

        if word in self.suggestions_map:
            suggestions, _ = self.suggestions_map[word]
            menu = QMenu(self)
            for suggestion in suggestions:
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
        self.setFixedSize(850, 550)

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

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.check_button)
        left_layout.addWidget(self.clear_button)
        left_layout.addWidget(self.text_edit)
        left_layout.addWidget(QLabel("Suggestions"))
        left_layout.addWidget(self.result_list)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Dictionary"))
        right_layout.addWidget(self.search_input)
        right_layout.addWidget(self.dictionary_list)

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

    def clear_text(self):
        self.text_edit.clear()
        self.result_list.clear()

    def check_text(self):
        self.result_list.clear()
        text = self.text_edit.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "Empty", "Please enter some text.")
            return

        errors = check_text_nlp(text)
        self.highlight_errors(errors)
        self.text_edit.set_suggestions_map(errors)

        if not errors:
            self.result_list.addItem("No errors found.")
        else:
            for word, (suggestions, err_type) in errors.items():
                self.result_list.addItem(f"{word} ({err_type}) ➜ {', '.join(suggestions)}")

    def highlight_errors(self, errors):
        cursor = self.text_edit.textCursor()
        cursor.select(QTextCursor.Document)
        cursor.setCharFormat(QTextCharFormat())

        text = self.text_edit.toPlainText()
        text_lower = text.lower()

        for word, (_, err_type) in errors.items():
            fmt = QTextCharFormat()
            fmt.setUnderlineStyle(QTextCharFormat.SpellCheckUnderline)
            if err_type == "non-word":
                fmt.setUnderlineColor(QColor("red"))
            elif err_type == "real-word":
                fmt.setUnderlineColor(QColor("blue"))
            elif err_type == "grammar":
                fmt.setUnderlineColor(QColor("green"))

            start = 0
            while True:
                start = text_lower.find(word, start)
                if start == -1:
                    break
                cursor.setPosition(start)
                cursor.setPosition(start + len(word), QTextCursor.KeepAnchor)
                cursor.setCharFormat(fmt)
                start += len(word)

    def search_dictionary(self):
        query = self.search_input.text().lower()
        self.dictionary_list.clear()
        filtered = [word for word in self.dictionary_words if query in word]
        self.dictionary_list.addItems(filtered)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpellCheckerApp()
    window.show()
    sys.exit(app.exec())
