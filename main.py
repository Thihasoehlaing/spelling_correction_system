import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QTextEdit, QPushButton, QLabel,
    QListWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox, QMenu
)
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor, QAction
from PySide6.QtCore import Qt

import nltk
from nltk.corpus import brown, words
from nltk import bigrams, trigrams
from spellchecker import SpellChecker
from collections import Counter
import spacy

# Download required NLTK data
nltk.download('punkt')
nltk.download('brown')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

# NLP setup
nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()

# Corpus prep
corpus_words = [w.lower() for w in brown.words() if w.isalpha()]
dictionary = set(words.words()) | set(corpus_words)
bigrams_c = Counter(bigrams(corpus_words))
trigrams_c = Counter(trigrams(corpus_words))

def check_text_advanced(text):
    doc = nlp(text)
    tokens = [t for t in doc if t.is_alpha]
    lower_tokens = [t.text.lower() for t in tokens]
    errors = {}

    # === Real-word misuse detection ===
    for i in range(1, len(tokens) - 1):  # CENTERED on actual_token
        w1, w2, w3 = lower_tokens[i - 1], lower_tokens[i], lower_tokens[i + 1]
        actual_token = tokens[i]
        actual_word = actual_token.text
        actual_pos = actual_token.pos_

        if not actual_word.isalpha() or w2 not in dictionary:
            continue

        if actual_pos in ["PRON", "PROPN", "DET", "NUM"]:
            continue

        trigram_count_actual = trigrams_c[(w1, w2, w3)]
        if trigram_count_actual > 0:
            continue

        candidates = spell.known(spell.edit_distance_1(w2))
        candidates = [w for w in candidates if w != w2 and w in dictionary]

        for candidate in candidates:
            trigram_count_candidate = trigrams_c[(w1, candidate, w3)]
            if trigram_count_candidate >= 1:
                cand_doc = nlp(candidate)
                if cand_doc and cand_doc[0].pos_ == actual_pos:
                    if cand_doc[0].lemma_ == actual_token.lemma_:
                        continue
                    errors[actual_word.lower()] = ([candidate], "real-word")
                    break



    # === Non-word detection ===
    for token in tokens:
        word = token.text
        lw = word.lower()
        if lw not in dictionary and lw not in errors:
            suggestions = list(spell.candidates(lw))[:3]
            errors[lw] = (suggestions, "non-word")

    # === Grammar rule detection (subject-verb) ===
    for token in doc:
        word = token.text
        lw = word.lower()
        if not word.isalpha() or lw in errors:
            continue

        if token.tag_ in ["VBP", "VBZ", "VBD", "VB"] and token.head.pos_ in ["NOUN", "PRON"]:
            subj = token.head.text.lower()
            if subj in ["he", "she", "it"] and token.tag_ == "VBP":
                errors[lw] = ([token.lemma_ + 's'], "grammar")
            elif subj in ["they", "we", "i"] and token.tag_ == "VBZ":
                errors[lw] = ([token.lemma_], "grammar")

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
        cursor.setCharFormat(QTextCharFormat())  # Clear formatting

        text_original = self.text_edit.toPlainText()
        text_lower = text_original.lower()

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
                start = text_lower.find(word, start)
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

# Run the app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpellCheckerApp()
    window.show()
    sys.exit(app.exec())
