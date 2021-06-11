from nltk.tokenize.treebank import TreebankWordDetokenizer

from typing import List

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from src.gui.Ui_MainWindow import Ui_MainWindow
from src.model.Seq2SeqModel import Seq2SeqModel
from src.model.generate_text import generate_sentence
from src.model.load_model import load_model
from src.model.params import params
from src.vocab.vocab_loader import load_vocab

from definitions import MODEL_PATH, VOCAB_PATH


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.generate_button.clicked.connect(self.generate_button_clicked)
        self.set_validation()

        self.vocab = load_vocab(VOCAB_PATH)
        self.model = Seq2SeqModel(**params)
        load_model(self.model, MODEL_PATH)

    def generate_button_clicked(self):
        if self.is_inputs_empty():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle('Ошибка')
            msg.setText('Поля ввода не должны быть пустыми!')
            msg.exec()
            return

        sentence = self.lemm_input_line.text()
        processed_input_sentence = self.preprocess_input_sentence(sentence)

        nsubj = self.nsubj_input_line.text().lower()
        gender_idx = self.gender_input_comboBox.currentIndex()
        tense_idx = self.tense_input_comboBox.currentIndex()

        generated_sentence = generate_sentence(self.model,
                                               processed_input_sentence,
                                               (nsubj, gender_idx, tense_idx),
                                               self.vocab)

        processed_generated_sentence = self.process_generated_sentence(generated_sentence)

        self.delemmatized_output_line.setText(processed_generated_sentence)

    def is_inputs_empty(self) -> bool:
        inputs = [self.lemm_input_line, self.nsubj_input_line]

        is_empty = any(input.text() == '' for input in inputs)
        return is_empty

    def preprocess_input_sentence(self, sentence: str):
        sentence = sentence.lower()
        ending_punctuation = '.!?'
        if sentence[-1] not in ending_punctuation:
            sentence += '.'
        return sentence

    def process_generated_sentence(self, tokenized_sentence: List[str]):
        first = [tokenized_sentence[0][0].upper() + tokenized_sentence[0][1:]]
        tokenized_sentence = first + tokenized_sentence[1:]
        detokenized_sentence = TreebankWordDetokenizer().detokenize(tokenized_sentence)

        return detokenized_sentence

    def set_validation(self):
        lemm_input_regex = QtCore.QRegExp(r'[^A-Za-z\s][^A-Za-z]*')
        nsubj_regex = QtCore.QRegExp(r'[^A-Za-z0-9 ]*')
        lemm_input_validator = QtGui.QRegExpValidator(lemm_input_regex)
        nsubj_validator = QtGui.QRegExpValidator(nsubj_regex)

        self.lemm_input_line.setValidator(lemm_input_validator)
        self.nsubj_input_line.setValidator(nsubj_validator)
