import sys
import os


def qrc_to_py():
    command = r'pyrcc5'
    source = r'./resources/resources.qrc'
    target = r'-o ./src/gui/resources_rc.py'
    os.system(rf'{command} {source} {target}')
    print('qrc успешно собран')


def ui_to_py():
    command = r'pyuic5'
    source = r'./ui/design.ui'
    target = r'-o ./src/gui/Ui_MainWindow.py'
    os.system(rf'{command} {source} {target}')

    with open(target[3:], 'r') as f:
        lines = f.readlines()

    lines[-1] = r'from src.gui import resources_rc'

    with open(target[3:], 'w') as f:
        f.writelines(lines)

    print('UI успешно собран')


def build_exe():
    command = r'pyinstaller'
    source = r'./src/application.py'
    flags = r'--onefile'
    name = r'--name sentence_delemmatizer'
    os.system(rf'{command} {source} {flags} {name}')


def main(argv):
    if len(argv) != 2:
        print('Необходимо указать аргумент\n'
              'Доступные варианты:\n'
              '\tbuild.py qrc\n'
              '\tbuild.py ui\n'
              '\tbuild.py exe\n')
        return

    actions = {
        'qrc': qrc_to_py,
        'ui': ui_to_py,
        'exe': build_exe
    }

    action = actions.get(argv[1],
                         lambda: print('Указан неверный аргумент, используйте один из следующих: (qrc, ui, exe)'))
    action()


if __name__ == '__main__':
    main(sys.argv)
