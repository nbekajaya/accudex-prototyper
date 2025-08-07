import sys
from PySide6 import QtCore, QtWidgets, QtGui

class FirstWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.hello = 'Test that shit out yo!'

        # Note how we don't pass in any parent during element stuff
        # We only do it in layout
        self.button = QtWidgets.QPushButton('Click me!')
        self.text = QtWidgets.QLabel('Text me!',
                                     alignment=QtCore.Qt.AlignCenter)
        
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.button)

        # Defines button push callback
        self.button.clicked.connect(self.magic)

    @QtCore.Slot()
    def magic(self):
        self.text.setText(self.hello)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = FirstWidget()
    widget.resize(800,600)
    widget.show()

    sys.exit(app.exec())

