from datetime import datetime

MOVE = 'MOVE'
APPEND = 'APPEND'
BACKSPACE = 'BACKSPACE'
SELECT = 'SELECT'
COPY = 'COPY'
PASTE = 'PASTE'
UNDO = 'UNDO'
REDO = 'REDO'
OPEN = 'OPEN'
CLOSE = 'CLOSE'


class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if len(self.stack) == 0:
            return None
        return self.stack.pop()

    def top(self):
        return self.stack[len(self.stack) - 1]

    def values(self):
        return self.stack


class Document:
    def __init__(self, name):
        self.name = name
        self.global_text = ''
        self.current_cursor = 0
        self.selected_left = None
        self.selected_right = None

        self.undo_stack = Stack()
        self.redo_stack = Stack()


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()


class Texteditor:
    document = Stack()
    current_document = None
    operations = []
    returns = []
    copied_value = ''

    def undo_push(self):
        self.current_document.undo_stack.push([self.current_document.global_text, self.current_document.current_cursor,
                                               self.current_document.selected_left, self.current_document.selected_right
                                               ])

    def redo_push(self):
        self.current_document.redo_stack.push([self.current_document.global_text, self.current_document.current_cursor,
                                               self.current_document.selected_left, self.current_document.selected_right
                                               ])

    def append(self, text: str):
        self.undo_push()
        if self.current_document.selected_left is not None and self.current_document.selected_right is not None:
            left_str = self.current_document.global_text[: self.current_document.selected_left]
            right_str = self.current_document.global_text[self.current_document.selected_right:]
        else:
            left_str = self.current_document.global_text[: self.current_document.current_cursor]
            right_str = self.current_document.global_text[self.current_document.current_cursor:]
        new_str = ''.join([left_str, text, right_str])
        self.current_document.global_text = new_str
        self.current_document.current_cursor = len(left_str) + len(text)
        self.current_document.selected_left = None
        self.current_document.selected_right = None
        self.current_document.redo_stack = Stack()
        self.redo_push()

    def move(self, cursor: int):
        if cursor <= len(self.current_document.global_text):
            self.current_document.current_cursor = cursor
        elif cursor > len(self.current_document.global_text):
            self.current_document.current_cursor = len(self.current_document.global_text) - 1
        elif cursor < 0:
            self.current_document.current_cursor = 0
        self.current_document.selected_left = None
        self.current_document.selected_right = None

    def backspace(self):
        self.undo_push()
        if self.current_document.selected_left is not None and self.current_document.selected_right is not None:
            left_str = self.current_document.global_text[: self.current_document.selected_left]
            right_str = self.current_document.global_text[self.current_document.selected_right:]
        else:
            left_str = self.current_document.global_text[: self.current_document.current_cursor - 1]
            right_str = self.current_document.global_text[self.current_document.current_cursor:]
        new_str = ''.join([left_str, '', right_str])
        self.current_document.global_text = new_str
        self.current_document.current_cursor = len(left_str)
        self.current_document.selected_left = None
        self.current_document.selected_right = None
        self.current_document.redo_stack = Stack()
        self.redo_push()

    def select(self, start: int, end: int):
        if start < end:
            self.current_document.selected_left = start
            self.current_document.selected_right = end

    def copy(self):
        self.copied_value = self.current_document.global_text[
                            self.current_document.selected_left: self.current_document.selected_right]

    def paste(self):
        self.undo_push()
        if self.current_document.selected_left is not None and self.current_document.selected_right is not None:
            left_str = self.current_document.global_text[: self.current_document.selected_left]
            right_str = self.current_document.global_text[self.current_document.selected_right:]
        else:
            left_str = self.current_document.global_text[: self.current_document.current_cursor]
            right_str = self.current_document.global_text[self.current_document.current_cursor:]
        new_str = ''.join([left_str, self.copied_value, right_str])
        self.current_document.global_text = new_str
        self.current_document.current_cursor = len(left_str) + len(self.copied_value)
        self.current_document.selected_left = None
        self.current_document.selected_right = None
        self.current_document.redo_stack = Stack()
        self.redo_push()

    def undo(self):
        undo_values = self.current_document.undo_stack.pop()
        if undo_values is not None:
            self.redo_push()
            self.current_document.global_text = undo_values[0]
            self.current_document.current_cursor = undo_values[1]
            self.current_document.selected_left = undo_values[2]
            self.current_document.selected_right = undo_values[3]

    def redo(self):
        redo_values = self.current_document.redo_stack.pop()
        if redo_values is not None:
            self.undo_push()
            self.current_document.global_text = redo_values[0]
            self.current_document.current_cursor = redo_values[1]
            self.current_document.selected_left = redo_values[2]
            self.current_document.selected_right = redo_values[3]

    def open(self, param):
        exist_flag = False
        doc = Document(param)
        for e in self.document.values():
            if e.name == param:
                # ignore
                exist_flag = True
                doc = e
        if exist_flag is False:
            self.document.push(doc)
        self.current_document = doc

    def close(self):
        self.document.pop()
        self.current_document = self.document.top()

    def print_text(self):
        print(self.current_document.global_text)

    def get_cursor(self):
        print(self.current_document.current_cursor)

    def command(self, command_name: str, *params: str):
        query = [command_name]
        query += params
        if len(self.document.values()) == 0:
            now = datetime.now()
            self.open(Document(str(now)))
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
        elif command_name == OPEN:
            self.open(params[0])
        elif command_name == CLOSE:
            self.close()
        result = [self.current_document.global_text]
        self.operations.append(query)
        self.returns.append(result)

    def dump(self):
        print(self.operations)
        print(self.returns)


if __name__ == '__main__':
    editor = Texteditor()
    editor.command('OPEN', 'document1')
    editor.command('APPEND', 'Hello, world!')
    editor.command('SELECT', '7', '12')
    editor.command('COPY')
    editor.command('BACKSPACE')
    editor.command('OPEN', 'document2')
    editor.command('PASTE')
    editor.command('CLOSE')
    editor.command('UNDO')
    editor.command('OPEN', 'document2')
    editor.command('UNDO')
    editor.dump()
