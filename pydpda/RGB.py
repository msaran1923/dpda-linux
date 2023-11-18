class RGB:
    def __init__(self, red=0, green=0, blue=0):
        self.red = red
        self.green = green
        self.blue = blue

    def __eq__(self, source):
        return (
            isinstance(source, RGB) and
            self.red == source.red and
            self.green == source.green and
            self.blue == source.blue
        )

    def __ne__(self, source):
        return not self.__eq__(source)
