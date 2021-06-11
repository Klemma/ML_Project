import sys

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox

from Ui_MainWindow import Ui_MainWindow
from application.src.model.Seq2SeqModel import Seq2SeqModel
from application.src.model.generate_text import generate_sentence
from application.src.model.load_model import load_model
from application.src.model.params import params
from application.src.vocab.Vocab import Vocab
from application.src.vocab.load_vocabs import load_vocabs
from definitions import MODEL_PATH, VOCABS_PATH


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.generate_button.clicked.connect(self.generate_button_clicked)
        self.set_validation()

        self.lemm_vocab = None
        self.orig_vocab = None
        self.lemm_vocab, self.orig_vocab = load_vocabs(VOCABS_PATH)
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
                                               processed_input_sentence, [nsubj, gender_idx, tense_idx],
                                               self.lemm_vocab, self.orig_vocab)

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

    def process_generated_sentence(self, sentence: str):
        sentence = sentence[0].upper() + sentence[1:]
        ending_punctuation = '.!?'
        if sentence[-1] in ending_punctuation:
            # deleting whitespace before ending punctuation (.!?)
            sentence = sentence[:-2] + '' + sentence[-1]
        return sentence

    def set_validation(self):
        lemm_input_regex = QtCore.QRegExp(r'[^A-Za-z\s][^A-Za-z]*')
        nsubj_regex = QtCore.QRegExp(r'[^A-Za-z0-9 ]*')
        lemm_input_validator = QtGui.QRegExpValidator(lemm_input_regex)
        nsubj_validator = QtGui.QRegExpValidator(nsubj_regex)

        self.lemm_input_line.setValidator(lemm_input_validator)
        self.nsubj_input_line.setValidator(nsubj_validator)
