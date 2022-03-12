
MOVE = 'MOVE'
APPEND = 'APPEND'
BACKSPACE = 'BACKSPACE'
SELECT = 'SELECT'
COPY = 'COPY'
PASTE = 'PASTE'
UNDO = 'UNDO'
REDO = 'REDO'


class Stack:
    def __init__(self):
        self.stack = []
    def push(self, item):
        self.stack.append(item)
    def pop(self):
        if len(self.stack) == 0:
            return None
        return self.stack.pop()

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

    undo_stack = Stack()
    redo_stack = Stack()

    def append(self, text: str):
        self.undo_stack.push([self.global_text, self.current_cursor, self.selected_left, self.selected_right])
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
        self.redo_stack = Stack()
        self.redo_stack.push([self.global_text, self.current_cursor, self.selected_left, self.selected_right])

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
        self.undo_stack.push([self.global_text, self.current_cursor, self.selected_left, self.selected_right])
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
        self.redo_stack = Stack()
        self.redo_stack.push([self.global_text, self.current_cursor, self.selected_left, self.selected_right])

    def select(self, start: int, end: int):
        if start < end:
            self.selected_left = start
            self.selected_right = end

    def copy(self):
        self.copied_value = self.global_text[self.selected_left: self.selected_right]

    def paste(self):
        self.undo_stack.push([self.global_text, self.current_cursor, self.selected_left, self.selected_right])
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
        self.redo_stack = Stack()
        self.redo_stack.push([self.global_text, self.current_cursor, self.selected_left, self.selected_right])

    def undo(self):
        undo_values = self.undo_stack.pop()
        if undo_values is not None:
            self.redo_stack.push([self.global_text, self.current_cursor, self.selected_left, self.selected_right])
            self.global_text = undo_values[0]
            self.current_cursor = undo_values[1]
            self.selected_left = undo_values[2]
            self.selected_right = undo_values[3]
    def redo(self):
        redo_values = self.redo_stack.pop()
        if redo_values is not None:
            self.undo_stack.push([self.global_text, self.current_cursor, self.selected_left, self.selected_right])
            self.global_text = redo_values[0]
            self.current_cursor = redo_values[1]
            self.selected_left = redo_values[2]
            self.selected_right = redo_values[3]
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
        elif command_name == UNDO:
            self.undo()
        elif command_name == REDO:
            self.redo()
        result = [self.global_text]
        self.operations.append(query)
        self.returns.append(result)

    def dump(self):
        print(self.operations)
        print(self.returns)


if __name__ == '__main__':
    editor = Texteditor()
    editor.command('APPEND', 'Hello, world!')
    editor.command('SELECT', '7', '12')
    editor.command('BACKSPACE')
    editor.command('UNDO')
    # editor.command('APPEND', 'you')
    editor.command('REDO')
    editor.command('UNDO')
    editor.command('APPEND', 'you')
    editor.command('REDO')
    # editor.command('APPEND', 'Hello cruel world!')
    # editor.command('SELECT', '5', '11')
    # editor.command('APPEND', ',')
    editor.dump()
