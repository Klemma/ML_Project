import sys

from PyQt5.QtWidgets import QApplication

from src.gui.MainWindow import MainWindow

if __name__ == '__main__':
    application = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(application.exec_())
