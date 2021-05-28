import sys

from PyQt5.QtWidgets import QMainWindow, QApplication
from Ui_MainWindow import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)


if __name__ == '__main__':
    application = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()

    sys.exit(application.exec_())
