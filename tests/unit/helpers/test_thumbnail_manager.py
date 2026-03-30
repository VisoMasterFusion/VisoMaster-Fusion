"""
TM-* tests for app.helpers.miscellaneous.ThumbnailManager
"""

import os
import numpy as np
import pytest
import cv2
from app.helpers.miscellaneous import ThumbnailManager


@pytest.fixture
def manager(tmp_path):
    """ThumbnailManager rooted in a fresh temp dir."""
    thumbdir = str(tmp_path / ".thumbnails")
    return ThumbnailManager(thumbnail_dir=thumbdir)


@pytest.fixture
def fake_media_file(tmp_path):
    """Write a tiny PNG file so os.path.getsize() returns a real value."""
    p = tmp_path / "sample.png"
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(p), img)
    return str(p)


@pytest.fixture
def fake_media_file2(tmp_path):
    """A different media file."""
    p = tmp_path / "other.png"
    img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    cv2.imwrite(str(p), img)
    return str(p)


# TM-01: same file always maps to same thumbnail path
def test_stable_path_for_same_file(manager, fake_media_file):
    png1, jpg1 = manager.get_thumbnail_path(fake_media_file)
    png2, jpg2 = manager.get_thumbnail_path(fake_media_file)
    assert png1 == png2
    assert jpg1 == jpg2


# TM-02: different files map to different paths
def test_different_files_different_paths(manager, fake_media_file, fake_media_file2):
    png1, _ = manager.get_thumbnail_path(fake_media_file)
    png2, _ = manager.get_thumbnail_path(fake_media_file2)
    assert png1 != png2


# TM-03: thumbnail directory is created on init
def test_directory_created_on_init(tmp_path):
    thumbdir = str(tmp_path / "nested" / ".thumbs")
    assert not os.path.exists(thumbdir)
    ThumbnailManager(thumbnail_dir=thumbdir)
    assert os.path.isdir(thumbdir)


# TM-04: creating a thumbnail produces a non-empty file on disk
def test_create_thumbnail_writes_file(manager, fake_media_file):
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    manager.create_thumbnail(frame, fake_media_file)
    png_path, jpg_path = manager.get_thumbnail_path(fake_media_file)
    exists = os.path.exists(png_path) or os.path.exists(jpg_path)
    assert exists
    found_path = png_path if os.path.exists(png_path) else jpg_path
    assert os.path.getsize(found_path) > 0


# TM-05: find_existing_thumbnail returns None before thumbnail is created
def test_find_existing_returns_none_before_creation(manager, fake_media_file):
    result = manager.find_existing_thumbnail(fake_media_file)
    assert result is None


# TM-06: find_existing_thumbnail returns path after thumbnail is created
def test_find_existing_returns_path_after_creation(manager, fake_media_file):
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    manager.create_thumbnail(frame, fake_media_file)
    result = manager.find_existing_thumbnail(fake_media_file)
    assert result is not None
    assert os.path.exists(result)


# TM-07: grayscale frame does not crash thumbnail creation
def test_create_thumbnail_grayscale_frame(manager, fake_media_file):
    gray = np.zeros((200, 200), dtype=np.uint8)
    manager.create_thumbnail(gray, fake_media_file)
    png_path, jpg_path = manager.get_thumbnail_path(fake_media_file)
    assert os.path.exists(png_path) or os.path.exists(jpg_path)


# TM-08: RGBA frame does not crash thumbnail creation
def test_create_thumbnail_rgba_frame(manager, fake_media_file):
    rgba = np.zeros((200, 200, 4), dtype=np.uint8)
    manager.create_thumbnail(rgba, fake_media_file)
    png_path, jpg_path = manager.get_thumbnail_path(fake_media_file)
    assert os.path.exists(png_path) or os.path.exists(jpg_path)
