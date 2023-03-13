import sys
from PyQt6.QtWidgets import QApplication, QLabel, QWidget
app=QApplication([])
window = QWidget()
window.setWindowTitle("PyQt App")
window.setGeometry(100, 100, 280, 80)
helloMsg = QLabel("<h1>Hello, World!</h1>", parent=window)
helloMsg.move(60, 15)
# 4. Show your application's GUI
window.show()

# 5. Run your application's event loop
sys.exit(app.exec())
