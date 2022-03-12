
MOVE = 'MOVE'
APPEND = 'APPEND'
BACKSPACE = 'BACKSPACE'
SELECT = 'SELECT'
COPY = 'COPY'
PASTE = 'PASTE'

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
    selected_left = None
    selected_right = None
    copied_value = ''

    def append(self, text: str):
        if self.selected_left is not None and self.selected_right is not None:
            left_str = self.global_text[: self.selected_left]
            right_str = self.global_text[self.selected_right:]
        else:
            left_str = self.global_text[: self.current_cursor]
            right_str = self.global_text[self.current_cursor:]
        new_str = ''.join([left_str, text, right_str])
        self.global_text = new_str
        self.current_cursor = len(left_str) + len(text)
        self.selected_left = None
        self.selected_right = None

    def move(self, cursor: int):
        if cursor <= len(self.global_text):
            self.current_cursor = cursor
        elif cursor > len(self.global_text):
            self.current_cursor = len(self.global_text) - 1
        elif cursor < 0:
            self.current_cursor = 0
        self.selected_left = None
        self.selected_right = None

    def backspace(self):
        if self.selected_left is not None and self.selected_right is not None:
            left_str = self.global_text[: self.selected_left]
            right_str = self.global_text[self.selected_right:]
        else:
            left_str = self.global_text[: self.current_cursor - 1]
            right_str = self.global_text[self.current_cursor:]
        new_str = ''.join([left_str, '', right_str])
        self.global_text = new_str
        self.current_cursor = len(left_str)
        self.selected_left = None
        self.selected_right = None

    def select(self, start: int, end: int):
        if start < end:
            self.selected_left = start
            self.selected_right = end

    def copy(self):
        self.copied_value =  self.global_text[self.selected_left: self.selected_right]

    def paste(self):
        if self.selected_left is not None and self.selected_right is not None:
            left_str = self.global_text[: self.selected_left]
            right_str = self.global_text[self.selected_right:]
        else:
            left_str = self.global_text[: self.current_cursor]
            right_str = self.global_text[self.current_cursor:]
        new_str = ''.join([left_str, self.copied_value, right_str])
        self.global_text = new_str
        self.current_cursor = len(left_str) + len(self.copied_value)
        self.selected_left = None
        self.selected_right = None
        print(self.current_cursor)

    def print_text(self):
        print(self.global_text)

    def get_cursor(self):
        print(self.current_cursor)

    def command(self, command_name: str, *params: str):
        query = [command_name]
        query += params
        if command_name == MOVE:
            param = params[0]
            if is_integer(param):
                int_param = int(param)
                self.move(int_param)
        elif command_name == APPEND:
            if params[0] is not None:
                self.append(params[0])
        elif command_name == BACKSPACE:
            self.backspace()
        elif command_name == SELECT:
            if is_integer(params[0]) and is_integer(params[1]):
                self.select(int(params[0]), int(params[1]))
        elif command_name == COPY:
            self.copy()
        elif command_name == PASTE:
            self.paste()
        result = [self.global_text]
        self.operations.append(query)
        self.returns.append(result)

    def dump(self):
        print(self.operations)
        print(self.returns)


if __name__ == '__main__':
    editor = Texteditor()
    editor.command('APPEND', 'Hello, world!')
    editor.command('SELECT', '5', '12')
    editor.command('COPY')
    editor.command('MOVE', '12')
    editor.command('PASTE')
    editor.command('PASTE')
    # editor.command('APPEND', 'Hello cruel world!')
    # editor.command('SELECT', '5', '11')
    # editor.command('APPEND', ',')
    editor.dump()
