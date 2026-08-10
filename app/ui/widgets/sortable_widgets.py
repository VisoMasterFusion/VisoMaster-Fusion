from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from functools import total_ordering
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PySide6 import QtCore, QtWidgets

from app.helpers.miscellaneous import MediaMetadata, get_file_type

if TYPE_CHECKING:
    from app.ui.widgets.widget_components import CardButton


class SortableDataRole(enum.IntEnum):
    """Item data roles, based off Qt.ItemDataRole.UserRole (0x100) to stay clear
    of the roles Qt reserves for itself."""

    FileInfoRole = 0x180  # 384
    ImageDimensionRole = 0x181  # 385
    CardButtonRole = 0x182  # 386
    FileTypeRole = 0x183  # 387
    MediaMetadataRole = 0x184  # 388


class SortingMode(enum.IntEnum):
    FileBirthDate = 0x140
    FileName = 0x141
    FileSize = 0x143
    ImageDimension = 0x144
    ImagePixels = 0x145
    VideoLength = 0x146

    def to_id(self) -> str:
        return self.name


initialize_combobox_items = [
    ("Sort by Name", SortingMode.FileName),
    ("Sort by File Date", SortingMode.FileBirthDate),
    ("Sort by File Size", SortingMode.FileSize),
    ("Sort by Dimensions", SortingMode.ImageDimension),
    ("Sort by Total Pixels", SortingMode.ImagePixels),
    ("Sort by Frame Length", SortingMode.VideoLength),
]

_FILE_MODES = frozenset(
    {SortingMode.FileBirthDate, SortingMode.FileName, SortingMode.FileSize}
)


@total_ordering
@dataclass(frozen=True)
class SortableImageDimension:
    width: int
    height: int

    def size(self) -> QtCore.QSize:
        return QtCore.QSize(self.width, self.height)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def sort_key(self) -> tuple[int, int]:
        return self.width, self.height

    def contains(self, minimum: "SortableImageDimension") -> bool:
        """True when this dimension is at least as large as minimum on both axes."""
        return self.width >= minimum.width and self.height >= minimum.height

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SortableImageDimension):
            return NotImplemented
        return self.sort_key == other.sort_key

    def __hash__(self) -> int:
        return hash(self.sort_key)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SortableImageDimension):
            return NotImplemented
        return self.sort_key < other.sort_key

    # PARSERS

    @classmethod
    def from_string(cls, value: str) -> "SortableImageDimension":
        match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", value)
        if match is None:
            raise ValueError(f"Not a WIDTHxHEIGHT dimension: {value!r}")
        return cls(int(match.group(1)), int(match.group(2)))

    @classmethod
    def from_metadata(
        cls, metadata: MediaMetadata | None
    ) -> Optional["SortableImageDimension"]:
        if metadata is None or not (metadata.width and metadata.height):
            return None
        return cls(int(metadata.width), int(metadata.height))


class SortableListWidget(QtWidgets.QListWidget):
    """QListWidget whose items sort themselves by the selected SortingMode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sorting_mode: SortingMode = SortingMode.FileName
        self._min_image_dimension: Optional[SortableImageDimension] = None

    def setSortingMode(self, sort_mode: SortingMode | None) -> None:
        if isinstance(sort_mode, SortingMode):
            self._sorting_mode = sort_mode

    def sortingMode(self) -> SortingMode:
        return self._sorting_mode

    def setMinImageDimension(self, imgdim: SortableImageDimension | None) -> None:
        self._min_image_dimension = imgdim

    def minImageDimension(self) -> Optional[SortableImageDimension]:
        return self._min_image_dimension


class TargetVideosListWidget(SortableListWidget):
    """Promoted class for MainWindow.ui's targetVideosList."""


class SortableListWidgetItem(QtWidgets.QListWidgetItem):
    def fileType(self) -> str:
        stored = self.data(SortableDataRole.FileTypeRole)
        if stored:
            return str(stored)
        fileinfo = self.fileInfo()
        if fileinfo is None:
            return ""
        return get_file_type(fileinfo.filePath()) or ""

    # DATA

    def cardButton(self) -> Optional["CardButton"]:
        return self.data(SortableDataRole.CardButtonRole)

    def setCardButton(self, card_button: "CardButton") -> None:
        self.setData(SortableDataRole.CardButtonRole, card_button)

    def fileInfo(self) -> Optional[QtCore.QFileInfo]:
        fileinfo = self.data(SortableDataRole.FileInfoRole)
        return fileinfo if isinstance(fileinfo, QtCore.QFileInfo) else None

    def setFileInfo(self, info: QtCore.QFileInfo) -> None:
        self.setData(SortableDataRole.FileInfoRole, info)

    def setFileType(self, file_type: str) -> None:
        self.setData(SortableDataRole.FileTypeRole, file_type)

    def imageDimension(self) -> Optional[SortableImageDimension]:
        imgdim = self.data(SortableDataRole.ImageDimensionRole)
        return imgdim if isinstance(imgdim, SortableImageDimension) else None

    def setImageDimension(self, imgdim: SortableImageDimension) -> None:
        self.setData(SortableDataRole.ImageDimensionRole, imgdim)

    def mediaMetadata(self) -> Optional[MediaMetadata]:
        metadata = self.data(SortableDataRole.MediaMetadataRole)
        return metadata if isinstance(metadata, MediaMetadata) else None

    def setMediaMetadata(self, metadata: MediaMetadata) -> None:
        self.setData(SortableDataRole.MediaMetadataRole, metadata)

    # COMPARATORS

    def _sort_key(self, sort_mode: SortingMode):
        """Value to order this item by, or None when it cannot be determined."""
        if sort_mode in _FILE_MODES:
            fileinfo = self.fileInfo()
            if fileinfo is None:
                return None
            if sort_mode == SortingMode.FileBirthDate:
                return fileinfo.birthTime()
            if sort_mode == SortingMode.FileName:
                return fileinfo.baseName().casefold()
            return fileinfo.size()

        if sort_mode == SortingMode.VideoLength:
            metadata = self.mediaMetadata()
            return None if metadata is None else metadata.total_frames

        imgdim = self.imageDimension()
        if imgdim is None:
            return None
        if sort_mode == SortingMode.ImageDimension:
            return imgdim.sort_key
        if sort_mode == SortingMode.ImagePixels:
            return imgdim.pixels
        return None

    def __lt__(self, other) -> bool:
        list_widget = self.listWidget()
        if not isinstance(list_widget, SortableListWidget):
            return False
        if not isinstance(other, SortableListWidgetItem):
            return False

        left = self._sort_key(list_widget.sortingMode())
        right = other._sort_key(list_widget.sortingMode())

        # Items that lack the data being sorted on (webcams have no file on disk)
        # sink below the ones that have it, rather than comparing equal to
        # everything, which would scramble the whole order.
        if left is None or right is None:
            return left is not None

        if left != right:
            return left < right

        # Qt's sort is not stable, so tie-break on the file name to keep items
        # with equal keys (same size, same dimensions, ...) in a predictable order.
        return self._tie_break_key() < other._tie_break_key()

    def _tie_break_key(self) -> str:
        fileinfo = self.fileInfo()
        return fileinfo.fileName().casefold() if fileinfo else ""

    # SERIALIZERS

    def to_tooltip(self) -> str:
        tipslist: list[str] = []
        fileinfo = self.fileInfo()
        if fileinfo:
            tipslist.append(f"FileName: {fileinfo.fileName()}")
            tipslist.append(f"FileType: {self.fileType()}")
            tipslist.append(
                f"FileDate: {fileinfo.birthTime().toLocalTime().toString('yyyy-MM-dd HH:mm:ss')}"
            )
            tipslist.append(
                f"FileSize: {QtCore.QLocale.system().formattedDataSize(fileinfo.size())}"
            )
            tipslist.append(f"Path: {Path(fileinfo.filePath()).parent}")

        image_dim = self.imageDimension()
        if image_dim:
            tipslist.append(f"Dimensions: {image_dim}")

        metadata = self.mediaMetadata()
        if metadata and self.fileType() == "video":
            tipslist.append(f"Length: {metadata.total_frames} frames")
            if metadata.frame_rate:
                tipslist.append(f"Frame Rate: {metadata.frame_rate:.3f}")
            if metadata.bitrate_kbits:
                tipslist.append(f"Bitrate: {metadata.bitrate_kbits:.0f} kBit/s")

        return "\n".join(tipslist)
