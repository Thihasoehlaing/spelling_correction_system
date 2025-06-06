import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QTextEdit, QPushButton,
    QLabel, QListWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QMessageBox, QMenu
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor, QAction

import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk import bigrams, trigrams, pos_tag
from spellchecker import SpellChecker
from collections import Counter

# First-time setup (run locally once):
# nltk.download('punkt')
# nltk.download('brown')
# nltk.download('averaged_perceptron_tagger')

spell = SpellChecker()
dictionary = set(spell.word_frequency.words())

brown_words = [w.lower() for w in brown.words() if w.isalpha()]
unigram_counts = Counter(brown_words)
bigram_counts = Counter(bigrams(brown_words))
trigram_counts = Counter(trigrams(brown_words))

def trigram_prob(w1, w2, w3):
    return trigram_counts[(w1, w2, w3)] / bigram_counts[(w1, w2)] if bigram_counts[(w1, w2)] > 0 else 0

confusion_sets = {
    'sea': ['see'],
    'their': ['there', 'they’re'],
    'your': ['you’re'],
    'its': ['it’s']
}

def grammar_errors(tokens):
    tags = pos_tag(tokens)
    tokens_lower = [w.lower() for w in tokens]
    errors = {}
    for i in range(len(tags) - 1):
        subject = tokens[i].lower()
        verb, verb_tag = tags[i + 1]

        if subject in ['he', 'she', 'it'] and verb_tag == 'VB':
            if verb.endswith('y') and verb[-2] not in 'aeiou':
                suggestion = [verb[:-1] + 'ies']
            elif verb in ['go', 'do']:
                suggestion = [verb + 'es']
            else:
                suggestion = [verb + 's']
            errors[verb.lower()] = (suggestion, 'grammar')
    return errors

def check_sentence(text):
    tokens = word_tokenize(text)
    tokens_lower = [w.lower() for w in tokens]
    errors = {}

    for i, word in enumerate(tokens_lower):
        if not word.isalpha():
            continue

        if word not in dictionary:
            suggestions = list(spell.candidates(word))
            errors[word] = (suggestions[:3], "non-word")
            continue

        if i >= 2 and word in confusion_sets:
            trigram_score = trigram_prob(tokens_lower[i - 2], tokens_lower[i - 1], word)
            if trigram_score < 0.001:
                errors[word] = (confusion_sets[word], "real-word")
                continue

        if i >= 2:
            w1, w2 = tokens_lower[i - 2], tokens_lower[i - 1]
            actual = trigram_prob(w1, w2, word)
            candidates = spell.candidates(word)
            ranked = sorted(
                [w for w in candidates if w != word and w in dictionary],
                key=lambda w: trigram_prob(w1, w2, w),
                reverse=True
            )
            if ranked:
                top_alt = ranked[0]
                top_prob = trigram_prob(w1, w2, top_alt)
                if top_prob > actual * 1.5:
                    errors[word] = ([top_alt], "real-word")

    errors.update(grammar_errors(tokens))
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
        self.setWindowTitle("Spelling Correction System (Final Grammar Enhanced)")
        self.setFixedSize(800, 500)

        self.text_edit = CustomTextEdit()
        self.text_edit.setPlaceholderText("Write something here (max 500 characters)...")
        self.text_edit.textChanged.connect(self.limit_text)

        self.check_button = QPushButton("Check Spelling")
        self.check_button.clicked.connect(self.check_spelling)

        self.clear_button = QPushButton("Clear Text")
        self.clear_button.clicked.connect(self.clear_text)

        self.result_label = QLabel("Suggestions:")
        self.result_list = QListWidget()

        self.dict_label = QLabel("Dictionary:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search word...")
        self.search_input.textChanged.connect(self.search_dictionary)

        self.dictionary_list = QListWidget()
        self.dictionary_words = sorted(dictionary)
        self.dictionary_list.addItems(self.dictionary_words)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.check_button)
        left_layout.addWidget(self.clear_button)
        left_layout.addWidget(self.text_edit)
        left_layout.addWidget(self.result_label)
        left_layout.addWidget(self.result_list)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.dict_label)
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

    def check_spelling(self):
        self.result_list.clear()
        text = self.text_edit.toPlainText()
        self.highlight_errors({}, clear=True)

        if not text.strip():
            QMessageBox.warning(self, "Empty Text", "Please write something first.")
            return

        results = check_sentence(text)
        self.text_edit.set_suggestions_map(results)
        self.highlight_errors(results)

        if not results:
            self.result_list.addItem("No spelling or grammar errors found.")
        else:
            for word, (suggestions, error_type) in results.items():
                self.result_list.addItem(f"{word} ({error_type}) ➜ {', '.join(suggestions)}")

    def highlight_errors(self, errors, clear=False):
        cursor = self.text_edit.textCursor()
        cursor.select(QTextCursor.Document)
        fmt_clear = QTextCharFormat()
        cursor.setCharFormat(fmt_clear)
        if clear:
            return

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
