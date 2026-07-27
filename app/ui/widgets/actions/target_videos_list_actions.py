from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

from app.ui.widgets import sortable_widgets
from app.ui.widgets.actions import filter_actions

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


def _set_value_blocking_signals(
    source_widget: QtWidgets.QSlider | QtWidgets.QSpinBox,
    dest_widget: QtWidgets.QSlider | QtWidgets.QSpinBox,
    value,
) -> None:
    """Mirror a value between a slider and its spin box without ping-ponging."""
    with QtCore.QSignalBlocker(source_widget):
        with QtCore.QSignalBlocker(dest_widget):
            dest_widget.setValue(value)


def _set_layout_widgets_visible(layout: QtWidgets.QLayout, checked: bool) -> None:
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is None:
            continue
        if widget := item.widget():
            widget.setVisible(checked)
        elif child_layout := item.layout():
            _set_layout_widgets_visible(child_layout, checked)


def update_target_videos_search_reset(main_window: "MainWindow", text: str) -> None:
    """Only offer the reset button while there is something to reset."""
    main_window.targetVideosSearchResetButton.setEnabled(bool(text and text.strip()))


def toggle_target_video_filters_sorting(main_window: "MainWindow") -> None:
    checked = main_window.targetVideosFilterMenuButton.isChecked()
    _set_layout_widgets_visible(main_window.horizontalLayout_9a, checked)
    _set_layout_widgets_visible(main_window.horizontalLayout_9b_gridLayout_3, checked)


def sort_target_videos(main_window: "MainWindow", *args, **kwargs) -> None:
    """Re-apply the current filters and sort order to the target media list."""
    list_widget = main_window.targetVideosList

    # Minimum image dimensions to keep an item visible.
    width = main_window.targetVideosWidthSlider.value() or 0
    height = main_window.targetVideosHeightSlider.value() or 0
    min_dimension = (
        sortable_widgets.SortableImageDimension(width, height)
        if (width or height)
        else None
    )
    list_widget.setMinImageDimension(min_dimension)

    sort_order = (
        QtCore.Qt.SortOrder.DescendingOrder
        if main_window.sortTargetVideosDirectionButton.isChecked()
        else QtCore.Qt.SortOrder.AscendingOrder
    )
    list_widget.setSortingMode(main_window.targetVideosSortComboBox.currentData())

    # Filter first so sorting only has to order the items that stay visible.
    filter_actions.filter_target_videos(main_window)
    list_widget.sortItems(sort_order)

    selected = main_window.selected_video_button
    selected_item = getattr(selected, "list_item", None) if selected else None
    if selected_item is not None:
        list_widget.scrollToItem(
            selected_item,
            QtWidgets.QAbstractItemView.ScrollHint.EnsureVisible,
        )
