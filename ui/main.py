from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FootAI")
        self.resize(1000, 700)

        tabs = QTabWidget()
        tabs.addTab(self._simple_tab("Matches"), "MATCHES")
        tabs.addTab(self._simple_tab("Analysis"), "ANALYSIS")
        tabs.addTab(self._simple_tab("Toto"), "TOTO")
        tabs.addTab(self._simple_tab("Training"), "TRAINING")
        self.setCentralWidget(tabs)

    def _simple_tab(self, text: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel(text))
        return widget


def run_ui() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
