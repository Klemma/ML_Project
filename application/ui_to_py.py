import os

if __name__ == '__main__':
    command = 'pyuic5'
    source = './ui/design.ui'
    target = './src/gui/Ui_MainWindow.py'
    flags = '-o'
    os.system(f'{command} {source} {flags} {target}')
