from Logger import Logger
class LoggerToConsole(Logger):
    def print(self, message):
        print(message, end='')

    def println(self, message):
        print(message)
