# record-results

record the program result in pycharm

代码段，放在程序运行之前

import sys
class Logger(object):

    def __init__(self, fileN="Default.log"):
    
        self.terminal = sys.stdout
        self.log = open(fileN, "a")
    def write(self, message):
    
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
    
        pass
sys.stdout = Logger("结果.txt")#可以自定义位置
