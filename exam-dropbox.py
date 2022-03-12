
MOVE = 'MOVE'
APPEND = 'APPEND'
BACKSPACE = 'BACKSPACE'


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()


class Texteditor:

    global_text = ''
    current_cursor = 0
    operations = []
    returns = []

    def append(self, text: str):
        left_str = self.global_text[: self.current_cursor]
        right_str = self.global_text[self.current_cursor:]
        new_str = ''.join([left_str, text, right_str])
        self.global_text = new_str
        self.current_cursor = len(left_str) + len(text)

    def move(self, cursor: int):
        if cursor <= len(self.global_text):
            self.current_cursor = cursor
        elif cursor > len(self.global_text):
            self.current_cursor = len(self.global_text) - 1
        elif cursor < 0:
            self.current_cursor = 0

    def backspace(self):
        left_str = self.global_text[: self.current_cursor]
        right_str = self.global_text[self.current_cursor:]
        left_str = left_str[:-1]
        new_str = ''.join([left_str, right_str])
        self.global_text = new_str
        if self.current_cursor > 0:
            self.current_cursor -= 1

    def print_text(self):
        print(self.global_text)

    def get_cursor(self):
        print(self.current_cursor)

    def command(self, command_name: str, param: str = None):
        query = [command_name, param]
        if command_name == MOVE:
            if is_integer(param):
                int_param = int(param)
                self.move(int_param)
        elif command_name == APPEND:
            if param is not None:
                self.append(param)
        elif command_name == BACKSPACE:
            self.backspace()
        result = [self.global_text]
        self.operations.append(query)
        self.returns.append(result)

    def dump(self):
        print(self.operations)
        print(self.returns)


if __name__ == '__main__':
    editor = Texteditor()
    editor.command('APPEND', 'Hey you')
    editor.command('MOVE', '3')
    editor.command('APPEND', ',')
    editor.command('BACKSPACE')
    editor.command('BACKSPACE')
    editor.command('MOVE', '1')
    editor.command('BACKSPACE')
    editor.command('BACKSPACE')
    editor.dump()
