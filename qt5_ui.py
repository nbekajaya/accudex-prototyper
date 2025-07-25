from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QBoxLayout,QLayout
from PyQt5.QtCore import QByteArray
from PyQt5.QtMultimedia import QCamera

app = QApplication([])
window = QWidget()
layout = QBoxLayout(QBoxLayout.TopToBottom)
layout.addWidget(QPushButton('Top'))
layout.addWidget(QPushButton('Bottom'))
window.setLayout(layout)
window.show()
app.exec()