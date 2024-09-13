import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resizable Window")
        self.label = QLabel(self)
        self.label.setText("Window size: (0, 0)")
        self.resize(400, 300)

    def resizeEvent(self, event):
        new_size = event.size()
        self.label.setText(f"Window size: ({new_size.width()}, {new_size.height()})")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
