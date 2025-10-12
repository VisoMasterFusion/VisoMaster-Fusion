import sys
import re
import json
import argparse
import copy

# MARKER: Import OrderedDict
from collections import OrderedDict
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QLineEdit,
    QLabel,
    QFrame,
    QScrollArea,
    QMenu,
    QInputDialog,
    QCheckBox,
    QComboBox,
    QGraphicsDropShadowEffect,
    QMenuBar,
)
from PySide6.QtCore import (
    Qt,
    QPoint,
    QRect,
    QThread,
    Signal,
    QObject,
    QMimeData,
    QEvent,
)
from PySide6.QtGui import QColor, QFont, QDrag, QPixmap, QAction

# Dark theme with red accent
DARK_BG = "#1A1A1A"
DARKER_BG = "#121212"
ITEM_BG = "#2C2C2C"
ITEM_HOVER_BG = "#3A3A3A"
BORDER_COLOR = "#444444"
TEXT_COLOR = "#E0E0E0"
ACCENT_COLOR = "#E74C3C"  # Red accent
ACCENT_HOVER = "#C0392B"  # Darker red
ACCENT_PRESSED = "#E74C3C"
SELECT_COLOR = "#E74C3C"  # Red for selection

# Layout constants
ITEMS_PER_STACK = 8
ENTRY_HEIGHT = 65
ENTRY_WIDTH = 220
STACK_SPACING = 12


class FileLoader(QObject):
    """Worker thread for loading and parsing files."""

    finished = Signal(object, str)  # Use object for OrderedDict

    def __init__(self, file_paths):
        super().__init__()
        self.file_paths = file_paths

    def run(self):
        # MARKER: Use OrderedDict to guarantee preservation of file order.
        embedding_data = OrderedDict()
        file_type = ""
        for file_path in self.file_paths:
            try:
                if file_path.endswith(".json"):
                    file_type = "json"
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = json.load(file)
                        for item in content:
                            if "name" in item and "embedding_store" in item:
                                embedding_data[item["name"]] = item["embedding_store"]
                else:
                    file_type = "txt"
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                        current_name = None
                        lines = content.splitlines()
                        for line in lines:
                            if line.startswith("Name:"):
                                current_name = line.split(":", 1)[1].strip()
                                embedding_data[current_name] = []
                            elif current_name and line.strip():
                                embedding_data[current_name].append(line.strip())
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
        self.finished.emit(embedding_data, file_type)


class EntryWidget(QFrame):
    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.name = name
        self.setup_ui()
        self.drag_start_pos = None

    def setup_ui(self):
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.setLineWidth(1)
        self.setFixedSize(ENTRY_WIDTH, ENTRY_HEIGHT)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(False)
        self.checkbox.stateChanged.connect(self.update_selection_style)
        layout.addWidget(self.checkbox, 0, Qt.AlignCenter)

        self.name_label = QLabel(self.name)
        self.name_label.setWordWrap(True)
        self.name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.name_label.setStyleSheet(f"""
            QLabel {{
                font-size: 15px;
                font-weight: 500;
                color: {TEXT_COLOR};
                padding: 4px;
            }}
        """)
        layout.addWidget(self.name_label, 1)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        self.update_selection_style()

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 160))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

    def update_selection_style(self):
        is_checked = self.checkbox.isChecked()
        self.setStyleSheet(f"""
            EntryWidget {{
                background-color: {ITEM_BG};
                border: 2px solid {SELECT_COLOR if is_checked else BORDER_COLOR};
                border-radius: 8px;
            }}
            EntryWidget:hover {{
                background-color: {ITEM_HOVER_BG};
                border: 2px solid {SELECT_COLOR if is_checked else ACCENT_COLOR};
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {BORDER_COLOR};
                border-radius: 5px;
            }}
            QCheckBox::indicator:checked {{
                background-color: {SELECT_COLOR};
                border: 2px solid {SELECT_COLOR};
            }}
        """)

    def mouseDoubleClickEvent(self, event):
        pos = event.position().toPoint()
        if self.name_label.geometry().contains(pos):
            self.start_rename()
        else:
            super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_pos = event.position()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton) or self.drag_start_pos is None:
            return
        if (
            event.position() - self.drag_start_pos
        ).manhattanLength() < QApplication.startDragDistance():
            return

        main_window = self.window()
        if isinstance(main_window, EmbeddingGUI):
            main_window.start_drag_operation()

    def mouseReleaseEvent(self, event):
        main_window = self.window()
        if event.button() == Qt.LeftButton:
            if isinstance(main_window, EmbeddingGUI):
                main_window.handle_entry_click(self, event)

    def show_context_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {ITEM_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                padding: 5px;
            }}
            QMenu::item:selected {{
                background-color: {ACCENT_COLOR};
            }}
        """)
        rename_action = menu.addAction("Rename")
        copy_action = menu.addAction("Copy")
        paste_action = menu.addAction("Paste")
        menu.addSeparator()
        delete_action = menu.addAction("Delete")

        action = menu.exec(self.mapToGlobal(pos))

        main_window = self.window()
        if not isinstance(main_window, EmbeddingGUI):
            return

        if action == rename_action:
            self.start_rename()
        elif action == copy_action:
            self.copy_to_clipboard()
        elif action == paste_action:
            main_window.paste_from_clipboard()
        elif action == delete_action:
            if not self.checkbox.isChecked():
                main_window.deselect_all()
                self.checkbox.setChecked(True)
            main_window.delete_selected_entries()

    def start_rename(self):
        text, ok = QInputDialog.getText(
            self, "Rename Entry", "Enter new name:", QLineEdit.Normal, self.name
        )
        if ok and text and text != self.name:
            main_window = self.window()
            if isinstance(main_window, EmbeddingGUI):
                main_window.rename_entry(self.name, text)

    def copy_to_clipboard(self):
        main_window = self.window()
        if not isinstance(main_window, EmbeddingGUI):
            return

        selected_items = main_window.get_selected_entries()
        if not selected_items:
            selected_items = [self]

        data_to_copy = []
        for item in selected_items:
            if item.name in main_window.embedding_data:
                data_to_copy.append(
                    {
                        "name": item.name,
                        "embedding_store": main_window.embedding_data[item.name],
                    }
                )

        if data_to_copy:
            QApplication.clipboard().setText(json.dumps(data_to_copy, indent=4))


class StacksWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.layout = QHBoxLayout(self)
        self.layout.setSpacing(STACK_SPACING)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.stacks = []
        self.drag_indicator = None

    def add_stack(self):
        stack_container = QWidget()
        stack_layout = QVBoxLayout(stack_container)
        stack_layout.setContentsMargins(0, 0, 0, 0)
        stack_layout.setSpacing(8)
        stack_layout.setAlignment(Qt.AlignTop)
        self.layout.addWidget(stack_container)
        self.stacks.append(stack_layout)
        return stack_layout

    def clear_entries(self):
        for stack in self.stacks:
            while stack.count():
                item = stack.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.stacks.clear()

    def get_all_widgets(self):
        widgets = []
        for stack in self.stacks:
            for i in range(stack.count()):
                widget = stack.itemAt(i).widget()
                if isinstance(widget, EntryWidget):
                    widgets.append(widget)
        return widgets

    def create_drag_indicator(self):
        if self.drag_indicator is None:
            self.drag_indicator = QFrame(self)
            self.drag_indicator.setFrameStyle(QFrame.NoFrame)
            self.drag_indicator.setStyleSheet(f"background-color: {ACCENT_COLOR};")
            self.drag_indicator.setFixedSize(ENTRY_WIDTH, 4)
            self.drag_indicator.hide()

    def update_drag_indicator(self, global_pos):
        self.create_drag_indicator()
        target_stack, index_in_stack = self.get_drop_location(global_pos)

        if target_stack:
            if index_in_stack < target_stack.count():
                widget = target_stack.itemAt(index_in_stack).widget()
                if widget:
                    pos = widget.mapToGlobal(widget.rect().topLeft())
                    indicator_pos = self.mapFromGlobal(pos)
                    indicator_pos.setY(indicator_pos.y() - STACK_SPACING // 2)
                    self.drag_indicator.move(indicator_pos)
                    self.drag_indicator.raise_()
                    self.drag_indicator.show()
                else:
                    self.drag_indicator.hide()
            elif target_stack.count() > 0:
                last_widget = target_stack.itemAt(target_stack.count() - 1).widget()
                if last_widget:
                    pos = last_widget.mapToGlobal(last_widget.rect().bottomLeft())
                    indicator_pos = self.mapFromGlobal(pos)
                    indicator_pos.setY(indicator_pos.y() - STACK_SPACING // 2)
                    self.drag_indicator.move(indicator_pos)
                    self.drag_indicator.raise_()
                    self.drag_indicator.show()
            else:
                pos = target_stack.parentWidget().mapToGlobal(QPoint(0, 0))
                indicator_pos = self.mapFromGlobal(pos)
                self.drag_indicator.move(indicator_pos)
                self.drag_indicator.raise_()
                self.drag_indicator.show()
        else:
            self.drag_indicator.hide()

    def dragEnterEvent(self, event):
        if event.mimeData().text() == "internal-move":
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        self.update_drag_indicator(self.mapToGlobal(event.position().toPoint()))
        event.accept()

    def dragLeaveEvent(self, event):
        if self.drag_indicator:
            self.drag_indicator.hide()

    def dropEvent(self, event):
        if self.drag_indicator:
            self.drag_indicator.hide()

        main_window = self.window()
        if not isinstance(main_window, EmbeddingGUI):
            return

        target_stack, index_in_stack = self.get_drop_location(
            self.mapToGlobal(event.position().toPoint())
        )

        if target_stack is None:
            return

        global_index = 0
        for stack in self.stacks:
            if stack == target_stack:
                global_index += index_in_stack
                break
            else:
                for i in range(stack.count()):
                    if isinstance(stack.itemAt(i).widget(), EntryWidget):
                        global_index += 1

        main_window.move_selected_entries(global_index)
        event.acceptProposedAction()

    def get_drop_location(self, global_pos):
        closest_stack = None
        min_dist = float("inf")

        for stack in self.stacks:
            stack_widget = stack.parentWidget()
            stack_center_x = stack_widget.mapToGlobal(stack_widget.rect().center()).x()
            dist = abs(stack_center_x - global_pos.x())

            global_top_left = stack_widget.mapToGlobal(stack_widget.rect().topLeft())
            global_bottom_right = stack_widget.mapToGlobal(
                stack_widget.rect().bottomRight()
            )
            global_stack_rect = QRect(global_top_left, global_bottom_right)

            if (
                global_pos.y() > global_stack_rect.top() - ENTRY_HEIGHT
                and global_pos.y() < global_stack_rect.bottom() + ENTRY_HEIGHT
            ):
                if dist < min_dist:
                    min_dist = dist
                    closest_stack = stack

        if not closest_stack:
            return None, -1

        for i in range(closest_stack.count()):
            widget = closest_stack.itemAt(i).widget()
            if isinstance(widget, EntryWidget):
                top_y = widget.mapToGlobal(widget.rect().topLeft()).y()
                drop_here_y = top_y + widget.height() // 2

                if global_pos.y() < drop_here_y:
                    return closest_stack, i

        entry_widget_count = sum(
            1
            for i in range(closest_stack.count())
            if isinstance(closest_stack.itemAt(i).widget(), EntryWidget)
        )
        return closest_stack, entry_widget_count


class EmbeddingGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Embedding Editor")
        # MARKER: Initialize with OrderedDict
        self.embedding_data = OrderedDict()
        self.original_embedding_data = OrderedDict()
        self.last_clicked_widget = None
        self.undo_stack = []
        self.redo_stack = []
        self.current_load_mode = "replace"
        self.setAcceptDrops(True)
        self.init_ui()

    def init_ui(self):
        self.setFont(QFont("Segoe UI", 10))

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._create_menu()
        main_layout.setMenuBar(self.menu_bar)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        self.load_button = QPushButton("Load File(s)")
        self.load_button.clicked.connect(self.load_files)
        top_layout.addWidget(self.load_button)

        self.additive_load_button = QPushButton("Load Additive")
        self.additive_load_button.clicked.connect(self.additive_load_files)
        top_layout.addWidget(self.additive_load_button)

        self.save_as_button = QPushButton("Save As")
        self.save_as_button.clicked.connect(self.save_as_file)
        top_layout.addWidget(self.save_as_button)

        self.save_button = QPushButton("Save Selected")
        self.save_button.clicked.connect(self.save_file)
        top_layout.addWidget(self.save_button)

        self.convert_button = QPushButton()
        self.convert_button.clicked.connect(self.convert_format)
        self.convert_button.setVisible(False)
        top_layout.addWidget(self.convert_button)

        self.model_combo = QComboBox()
        self.model_combo.addItems(
            ["Inswapper128ArcFace", "SimSwapArcFace", "GhostArcFace"]
        )
        self.model_combo.setVisible(False)
        top_layout.addWidget(self.model_combo)

        top_layout.addStretch()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search...")
        self.search_input.textChanged.connect(self.filter_entries)
        top_layout.addWidget(self.search_input)

        layout.addWidget(top_widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.installEventFilter(self)
        self.entries_widget = StacksWidget()
        self.scroll_area.setWidget(self.entries_widget)
        layout.addWidget(self.scroll_area)

        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(self.select_all)
        bottom_layout.addWidget(self.select_all_button)

        self.deselect_all_button = QPushButton("Deselect All")
        self.deselect_all_button.clicked.connect(self.deselect_all)
        bottom_layout.addWidget(self.deselect_all_button)

        bottom_layout.addStretch()

        sort_label = QLabel("Sort by:")
        bottom_layout.addWidget(sort_label)

        self.sorting_combo = QComboBox()
        self.sorting_combo.addItems(["Manual", "Original", "A-Z", "Z-A"])
        self.sorting_combo.currentIndexChanged.connect(self.apply_sorting)
        bottom_layout.addWidget(self.sorting_combo)

        layout.addWidget(bottom_widget)

        main_layout.addWidget(content_widget)
        self.setLayout(main_layout)
        self.resize(1400, 800)
        self.apply_styles()
        self.update_button_visibility(False)

    def _create_menu(self):
        self.menu_bar = QMenuBar(self)

        edit_menu = self.menu_bar.addMenu("&Edit")
        undo_action = QAction("Undo", self)
        undo_action.setShortcut(Qt.CTRL | Qt.Key_Z)
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut(Qt.CTRL | Qt.SHIFT | Qt.Key_Z)
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        paste_action = QAction("Paste", self)
        paste_action.setShortcut(Qt.CTRL | Qt.Key_V)
        paste_action.triggered.connect(self.paste_from_clipboard)
        edit_menu.addAction(paste_action)

        help_menu = self.menu_bar.addMenu("&Help")
        shortcuts_action = QAction("About Shortcuts", self)
        shortcuts_action.triggered.connect(self.show_help_dialog)
        help_menu.addAction(shortcuts_action)

    def eventFilter(self, source, event):
        if source is self.scroll_area and event.type() == QEvent.Type.Wheel:
            h_bar = self.scroll_area.horizontalScrollBar()
            vertical_delta = event.angleDelta().y()
            new_value = h_bar.value() - vertical_delta
            h_bar.setValue(new_value)
            return True
        return super().eventFilter(source, event)

    def apply_styles(self):
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {DARK_BG};
                color: {TEXT_COLOR};
                font-family: "Segoe UI";
            }}
            QPushButton {{
                background-color: {ACCENT_COLOR};
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: {ACCENT_HOVER};
            }}
            QPushButton:pressed {{
                background-color: {ACCENT_PRESSED};
            }}
            QLineEdit {{
                padding: 8px;
                border: 1px solid {BORDER_COLOR};
                border-radius: 5px;
                background-color: {ITEM_BG};
                font-size: 14px;
            }}
            QLineEdit:focus {{
                border: 1px solid {ACCENT_COLOR};
            }}
            QScrollArea {{
                border: none;
                background-color: {DARKER_BG};
            }}
            QScrollBar:horizontal, QScrollBar:vertical {{
                border: none;
                background: {DARKER_BG};
                height: 12px;
                width: 12px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal, QScrollBar::handle:vertical {{
                background: {ITEM_BG};
                min-width: 20px;
                min-height: 20px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal:hover, QScrollBar::handle:vertical:hover {{
                background: {ACCENT_COLOR};
            }}
            QComboBox {{
                background-color: {ITEM_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 5px;
                padding: 5px 10px;
            }}
            QComboBox:hover {{
                border-color: {ACCENT_COLOR};
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {ITEM_BG};
                border: 1px solid {BORDER_COLOR};
                selection-background-color: {ACCENT_COLOR};
            }}
            QMenuBar {{
                background-color: {DARKER_BG};
                color: {TEXT_COLOR};
            }}
            QMenuBar::item:selected {{
                background-color: {ACCENT_COLOR};
            }}
            QMenu {{
                 background-color: {DARKER_BG};
            }}
        """)

    def update_button_visibility(self, visible):
        is_data_loaded = bool(self.embedding_data)
        self.select_all_button.setVisible(visible)
        self.deselect_all_button.setVisible(visible)
        self.sorting_combo.setVisible(visible)
        self.save_button.setVisible(visible)
        self.save_as_button.setVisible(visible)
        self.search_input.setVisible(visible)
        self.additive_load_button.setVisible(is_data_loaded)

        if visible:
            self.convert_button.setVisible(True)
            if hasattr(self, "current_file_type"):
                if self.current_file_type == "txt":
                    self.convert_button.setText("Convert to Viso")
                    self.model_combo.setVisible(True)
                else:
                    self.convert_button.setText("Convert to Rope")
                    self.model_combo.setVisible(False)
        else:
            self.convert_button.setVisible(False)
            self.model_combo.setVisible(False)

    def keyPressEvent(self, event):
        modifiers = event.modifiers()
        if event.key() == Qt.Key_A and modifiers == Qt.ControlModifier:
            self.select_all()
        elif event.key() == Qt.Key_D and modifiers == Qt.ControlModifier:
            self.deselect_all()
        elif event.key() == Qt.Key_Delete:
            self.delete_selected_entries()
        else:
            super().keyPressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        file_paths = [url.toLocalFile() for url in urls if url.isLocalFile()]
        valid_paths = [path for path in file_paths if path.endswith((".json", ".txt"))]

        if not valid_paths:
            return

        if not self.embedding_data:
            self.start_file_loading_thread(valid_paths, "replace")
            return

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Load Files")
        msg_box.setText("How would you like to load the dropped files?")
        msg_box.setIcon(QMessageBox.Icon.Question)
        replace_button = msg_box.addButton(
            "Load (Replace)", QMessageBox.ButtonRole.ActionRole
        )
        additive_button = msg_box.addButton(
            "Load (Additive)", QMessageBox.ButtonRole.ActionRole
        )
        cancel_button = msg_box.addButton(QMessageBox.StandardButton.Cancel)
        msg_box.exec()

        clicked_button = msg_box.clickedButton()
        if clicked_button == replace_button:
            self.start_file_loading_thread(valid_paths, "replace")
        elif clicked_button == additive_button:
            self.start_file_loading_thread(valid_paths, "additive")

    def load_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Load Files", "", "JSON Files (*.json);;Text Files (*.txt)"
        )
        if file_paths:
            self.start_file_loading_thread(file_paths, "replace")

    def additive_load_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Load Additive", "", "JSON Files (*.json);;Text Files (*.txt)"
        )
        if file_paths:
            self.start_file_loading_thread(file_paths, "additive")

    def start_file_loading_thread(self, file_paths, mode):
        self.current_load_mode = mode
        self.thread = QThread()
        self.worker = FileLoader(file_paths)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_loading_finished)
        self.thread.start()
        self.load_button.setEnabled(False)
        self.additive_load_button.setEnabled(False)
        self.load_button.setText("Loading...")

    def on_loading_finished(self, new_data, file_type):
        if self.current_load_mode == "replace":
            self.embedding_data = new_data
            # MARKER: The loaded file's order is now the new "Original" order.
            self.original_embedding_data = copy.deepcopy(new_data)
            self.undo_stack.clear()
            self.redo_stack.clear()

        elif self.current_load_mode == "additive":
            self._save_state_for_undo()
            for name, data in new_data.items():
                if name not in self.embedding_data:
                    self.embedding_data[name] = data
                    self.original_embedding_data[name] = copy.deepcopy(data)

        self.current_file_type = file_type
        # Block signals to prevent apply_sorting from firing accidentally
        self.sorting_combo.blockSignals(True)
        self.sorting_combo.setCurrentText("Manual")
        self.sorting_combo.blockSignals(False)

        self.populate_entries()
        self.update_button_visibility(True)
        self.thread.quit()
        self.thread.wait()
        self.load_button.setEnabled(True)
        self.additive_load_button.setEnabled(True)
        self.load_button.setText("Load File(s)")

    def populate_entries(self):
        self.entries_widget.clear_entries()

        if not self.embedding_data:
            return

        total_items = len(self.embedding_data)
        num_stacks = (total_items + ITEMS_PER_STACK - 1) // ITEMS_PER_STACK
        if num_stacks == 0 and total_items > 0:
            num_stacks = 1

        names = list(self.embedding_data.keys())
        item_idx = 0
        for i in range(num_stacks):
            stack_layout = self.entries_widget.add_stack()
            for j in range(ITEMS_PER_STACK):
                if item_idx < total_items:
                    name = names[item_idx]
                    entry_widget = EntryWidget(name, stack_layout.parentWidget())
                    stack_layout.addWidget(entry_widget)
                    item_idx += 1
                else:
                    spacer = QWidget()
                    spacer.setFixedSize(ENTRY_WIDTH, ENTRY_HEIGHT)
                    stack_layout.addWidget(spacer)

    def filter_entries(self):
        search_text = self.search_input.text().lower()
        for widget in self.entries_widget.get_all_widgets():
            widget.setVisible(search_text in widget.name.lower())

    def rename_entry(self, old_name, new_name):
        if old_name in self.embedding_data and new_name not in self.embedding_data:
            self._save_state_for_undo()
            # MARKER: Rebuild as an OrderedDict to preserve order
            new_embedding_data = OrderedDict()
            for name, data in self.embedding_data.items():
                if name == old_name:
                    new_embedding_data[new_name] = data
                else:
                    new_embedding_data[name] = data
            self.embedding_data = new_embedding_data
            self.populate_entries()
            self.sorting_combo.setCurrentText("Manual")

    def get_selected_entries(self):
        return [
            widget
            for widget in self.entries_widget.get_all_widgets()
            if widget.checkbox.isChecked()
        ]

    def select_all(self):
        for widget in self.entries_widget.get_all_widgets():
            if widget.isVisible():
                widget.checkbox.setChecked(True)

    def deselect_all(self):
        for widget in self.entries_widget.get_all_widgets():
            widget.checkbox.setChecked(False)
        self.last_clicked_widget = None

    def natural_sort_key(self, s):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)
        ]

    def apply_sorting(self):
        mode = self.sorting_combo.currentText()
        if not self.embedding_data or mode == "Manual":
            return

        self._save_state_for_undo()

        if mode == "A-Z":
            sorted_names = sorted(self.embedding_data.keys(), key=self.natural_sort_key)
            # MARKER: Rebuild as an OrderedDict to preserve order
            self.embedding_data = OrderedDict(
                (name, self.embedding_data[name]) for name in sorted_names
            )
        elif mode == "Z-A":
            sorted_names = sorted(
                self.embedding_data.keys(), key=self.natural_sort_key, reverse=True
            )
            # MARKER: Rebuild as an OrderedDict to preserve order
            self.embedding_data = OrderedDict(
                (name, self.embedding_data[name]) for name in sorted_names
            )
        elif mode == "Original":
            original_order = list(self.original_embedding_data.keys())
            current_keys = set(self.embedding_data.keys())
            # MARKER: Rebuild as an OrderedDict to preserve order
            self.embedding_data = OrderedDict(
                (name, self.embedding_data[name])
                for name in original_order
                if name in current_keys
            )

        self.populate_entries()

    def save_file(self):
        selected_entries = self.get_selected_entries()
        if not selected_entries:
            QMessageBox.warning(self, "No Selection", "No entries selected to save.")
            return

        all_widgets = self.entries_widget.get_all_widgets()
        ordered_names_to_save = [w.name for w in all_widgets if w in selected_entries]

        # MARKER: Build the save data as an OrderedDict to be safe
        selected_data = OrderedDict(
            (name, self.embedding_data[name]) for name in ordered_names_to_save
        )

        default_filter = (
            "JSON Files (*.json)"
            if self.current_file_type == "json"
            else "Text Files (*.txt)"
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save File", "", f"{default_filter};;All Files (*)"
        )

        if file_path:
            self._write_to_file(file_path, selected_data)

    def save_as_file(self):
        if not self.embedding_data:
            QMessageBox.warning(self, "No Data", "There is no data to save.")
            return

        all_widgets = self.entries_widget.get_all_widgets()
        ordered_names_to_save = [w.name for w in all_widgets]

        # MARKER: Build the save data as an OrderedDict to be safe
        all_data = OrderedDict(
            (name, self.embedding_data[name]) for name in ordered_names_to_save
        )

        default_filter = (
            "JSON Files (*.json)"
            if self.current_file_type == "json"
            else "Text Files (*.txt)"
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save File As", "", f"{default_filter};;All Files (*)"
        )

        if file_path:
            self._write_to_file(file_path, all_data)

    def _write_to_file(self, file_path, data_to_save):
        try:
            if file_path.endswith(".json"):
                json_data = [
                    {"name": name, "embedding_store": values}
                    for name, values in data_to_save.items()
                ]
                with open(file_path, "w", encoding="utf-8") as file:
                    json.dump(json_data, file, indent=4)
            else:
                with open(file_path, "w", encoding="utf-8") as file:
                    for name, values in data_to_save.items():
                        file.write(f"Name: {name}\n")
                        for value in values:
                            file.write(f"{value}\n")
            QMessageBox.information(self, "Success", "File saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {e}")

    def delete_selected_entries(self):
        selected_widgets = self.get_selected_entries()
        if not selected_widgets:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete {len(selected_widgets)} selected item(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._save_state_for_undo()
            for widget in selected_widgets:
                if widget.name in self.embedding_data:
                    del self.embedding_data[widget.name]
            self.populate_entries()
            self.last_clicked_widget = None
            self.sorting_combo.setCurrentText("Manual")

    def handle_entry_click(self, widget, event):
        all_widgets = [
            w for w in self.entries_widget.get_all_widgets() if w.isVisible()
        ]
        modifiers = event.modifiers()
        is_shift = modifiers & Qt.ShiftModifier
        is_ctrl = modifiers & Qt.ControlModifier

        try:
            current_index = all_widgets.index(widget)
        except ValueError:
            return

        if is_shift and self.last_clicked_widget in all_widgets:
            try:
                last_index = all_widgets.index(self.last_clicked_widget)
                start, end = (
                    min(last_index, current_index),
                    max(last_index, current_index),
                )

                if not is_ctrl:
                    for i, w in enumerate(all_widgets):
                        if not (start <= i <= end):
                            w.checkbox.setChecked(False)

                for i in range(start, end + 1):
                    all_widgets[i].checkbox.setChecked(True)
            except (ValueError, IndexError):
                self.last_clicked_widget = widget

        elif is_ctrl:
            widget.checkbox.setChecked(not widget.checkbox.isChecked())
            self.last_clicked_widget = widget

        else:
            is_currently_checked = widget.checkbox.isChecked()
            self.deselect_all()
            widget.checkbox.setChecked(not is_currently_checked)
            self.last_clicked_widget = widget

    def paste_from_clipboard(self):
        clipboard_text = QApplication.clipboard().text()
        if not clipboard_text:
            return

        try:
            pasted_data = json.loads(clipboard_text)
            if not isinstance(pasted_data, list):
                raise ValueError("Clipboard data is not a list of embeddings.")

            self._save_state_for_undo()
            added_count = 0
            for item in pasted_data:
                if "name" in item and "embedding_store" in item:
                    name = item["name"]
                    if name in self.embedding_data:
                        base_name = name
                        i = 1
                        while name in self.embedding_data:
                            name = f"{base_name}_{i}"
                            i += 1

                    self.embedding_data[name] = item["embedding_store"]
                    added_count += 1

            if added_count > 0:
                self.populate_entries()
                self.sorting_combo.setCurrentText("Manual")
                QMessageBox.information(
                    self,
                    "Paste Successful",
                    f"{added_count} embedding(s) have been pasted.",
                )
            else:
                QMessageBox.warning(
                    self, "Paste", "No valid embeddings found in the clipboard."
                )

        except (json.JSONDecodeError, ValueError) as e:
            QMessageBox.critical(
                self, "Paste Error", f"Could not paste embeddings: {e}"
            )

    def convert_format(self):
        if not self.embedding_data:
            return
        self._save_state_for_undo()

        if self.current_file_type == "txt":
            recognizer_model = self.model_combo.currentText()
            # MARKER: Rebuild as an OrderedDict
            new_data = OrderedDict()
            for name, values in self.embedding_data.items():
                try:
                    float_values = [float(val) for val in values]
                    new_data[name] = {recognizer_model: float_values}
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Conversion Error",
                        f"Could not convert values for '{name}' to numbers. Skipping.",
                    )
                    continue
            self.embedding_data = new_data
            self.current_file_type = "json"
            QMessageBox.information(self, "Success", "Converted to Viso (JSON) Format")
        else:
            # MARKER: Rebuild as an OrderedDict
            new_data = OrderedDict()
            for name, data in self.embedding_data.items():
                if isinstance(data, dict) and data:
                    model_name = next(iter(data))
                    values = data[model_name]
                    str_values = [str(val) for val in values]
                    new_data[name] = str_values
                else:
                    new_data[name] = []
            self.embedding_data = new_data
            self.current_file_type = "txt"
            QMessageBox.information(self, "Success", "Converted to Rope (TXT) Format")

        self.populate_entries()
        self.update_button_visibility(True)
        self.sorting_combo.setCurrentText("Manual")

    def start_drag_operation(self):
        selected_widgets = self.get_selected_entries()
        if not selected_widgets:
            return

        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText("internal-move")
        drag.setMimeData(mime_data)

        pixmap = self.create_drag_pixmap(selected_widgets)
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(pixmap.width() // 4, pixmap.height() // 4))

        drag.exec(Qt.MoveAction)

    def create_drag_pixmap(self, widgets):
        num_widgets = len(widgets)
        preview = QFrame()
        preview.setFixedSize(ENTRY_WIDTH + 10, ENTRY_HEIGHT + 10)
        preview.setStyleSheet(f"""
            QFrame {{
                background-color: {ITEM_BG};
                border: 2px solid {ACCENT_COLOR};
                border-radius: 8px;
            }}
        """)

        label = QLabel(f"{num_widgets} item(s)", preview)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setGeometry(0, 0, ENTRY_WIDTH, ENTRY_HEIGHT)
        label.setStyleSheet(
            f"color: {TEXT_COLOR}; background: transparent; border: none; font-size: 16px; font-weight: bold;"
        )

        pixmap = QPixmap(preview.size())
        preview.render(pixmap)
        return pixmap

    def move_selected_entries(self, drop_index):
        self._save_state_for_undo()

        all_item_names = list(self.embedding_data.keys())
        selected_item_names = [w.name for w in self.get_selected_entries()]
        remaining_items = [
            name for name in all_item_names if name not in selected_item_names
        ]

        if drop_index > len(remaining_items):
            drop_index = len(remaining_items)

        new_order = (
            remaining_items[:drop_index]
            + selected_item_names
            + remaining_items[drop_index:]
        )

        # MARKER: Rebuild as an OrderedDict to preserve order
        self.embedding_data = OrderedDict(
            (name, self.embedding_data[name]) for name in new_order
        )

        self.populate_entries()
        for widget in self.entries_widget.get_all_widgets():
            if widget.name in selected_item_names:
                widget.checkbox.setChecked(True)
        self.sorting_combo.setCurrentText("Manual")

    def _save_state_for_undo(self):
        self.undo_stack.append(copy.deepcopy(self.embedding_data))
        self.redo_stack.clear()
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def _restore_state(self, data_to_restore):
        self.embedding_data = data_to_restore
        self.populate_entries()
        self.update_button_visibility(True)
        self.sorting_combo.setCurrentText("Manual")

    def undo(self):
        if not self.undo_stack:
            return
        self.redo_stack.append(copy.deepcopy(self.embedding_data))
        last_state = self.undo_stack.pop()
        self._restore_state(last_state)

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(copy.deepcopy(self.embedding_data))
        next_state = self.redo_stack.pop()
        self._restore_state(next_state)

    def show_help_dialog(self):
        shortcuts_text = """
        <h3>Keyboard Shortcuts</h3>
        <p><b>Ctrl + A</b>: Select all visible entries.</p>
        <p><b>Ctrl + D</b>: Deselect all entries.</p>
        <p><b>Ctrl + V</b>: Paste copied entries.</p>
        <p><b>Delete</b>: Delete all selected entries.</p>
        <p><b>Ctrl + Z</b>: Undo the last action.</p>
        <p><b>Ctrl + Shift + Z</b>: Redo the last undone action.</p>

        <h3>Mouse Controls</h3>
        <p><b>Click</b>: Select a single entry.</p>
        <p><b>Ctrl + Click</b>: Add or remove an entry from the selection.</p>
        <p><b>Shift + Click</b>: Select a range of entries.</p>
        <p><b>Mouse Wheel</b>: Scroll horizontally.</p>
        <p><b>Double-Click Name</b>: Rename an entry.</p>
        <p><b>Drag & Drop File(s)</b>: Load files into the editor.</p>
        <p><b>Drag & Drop Entry</b>: Reorder entries.</p>
        <p><b>Right-Click</b>: Open context menu (Rename, Copy, Paste, Delete).</p>
        """
        QMessageBox.information(self, "About Shortcuts", shortcuts_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Editor")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    gui = EmbeddingGUI()
    gui.showMaximized()
    sys.exit(app.exec())
