import re

from PyQt5.QtWidgets import QMainWindow, QMessageBox

from src.gui.Ui_MainWindow import Ui_MainWindow

from src.transforms.context_transforms import idx_to_gender, idx_to_tense, idx_to_number
from src.transforms.delemmatize_sentence import delemmatize_sentence

from src.model.Seq2SeqMbart import Seq2SeqMbart
from src.model.Tokenizer import TokenizerWrapper


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.generate_button.clicked.connect(self.generate_button_clicked)
        self.tokenizer = TokenizerWrapper()
        self.model = Seq2SeqMbart()

    def get_input_sentence(self) -> str:
        return self.lemm_input_line.text()

    def get_context(self) -> tuple:
        nsubj = self.nsubj_input_line.text()
        gender = idx_to_gender[self.gender_input_comboBox.currentIndex()]
        tense = idx_to_tense[self.tense_input_comboBox.currentIndex()]
        number = idx_to_number[self.number_input_comboBox.currentIndex()]

        context = (nsubj, gender, tense, number)
        return context

    def set_output_sentence(self, sentence: str) -> None:
        self.delemmatized_output_line.setText(sentence)

    def generate_button_clicked(self):
        if self.is_inputs_empty():
            QMessageBox(QMessageBox.Critical, 'Ошибка', 'Поля вводы не должны быть пустыми').exec_()
            return

        sentence = self.preprocess_input_sentence(self.get_input_sentence())
        context = self.get_context()

        generated_sentence = delemmatize_sentence(self.model, self.tokenizer, sentence, context)
        generated_sentence = self.process_generated_sentence(generated_sentence)

        self.set_output_sentence(generated_sentence)

    def is_inputs_empty(self) -> bool:
        is_empty = self.lemm_input_line.text() == '' or self.nsubj_input_line == ''
        return is_empty

    def preprocess_input_sentence(self, sentence: str) -> str:
        end_punctuation = ['.', '!', '?']
        if sentence[-1] not in end_punctuation:
            sentence += '.'
        # sentence = sentence.lower()
        return sentence

    def process_generated_sentence(self, sentence:str) -> str:
        sentence = re.sub(r'« [\w\s]+ »', lambda m: f'{m.group()[0]}{m.group()[2:-2]}{m.group()[-1]}', sentence)
        sentence = re.sub(r'" [\w\s]+ "', lambda m: f'{m.group()[0]}{m.group()[2:-2]}{m.group()[-1]}', sentence)
        sentence = re.sub(r'\w - \w', lambda m: f'{m.group()[0]}{m.group()[2]}{m.group()[-1]}', sentence)
        return sentence
