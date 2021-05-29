import os

if __name__ == '__main__':
    command = 'pyrcc5'
    source = './resources/resources.qrc'
    target = './src/gui/resources_rc.py'
    flags = '-o'
    os.system(f'{command} {source} {flags} {target}')
