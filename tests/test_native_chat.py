"""End-to-end checks for the browser-free chat renderer."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QTextDocument
from PyQt6.QtWidgets import QApplication

from core.native_chat import NativeRichLogWidget


def test_native_renderer_keeps_markdown_code_and_math_visible():
    app = QApplication.instance() or QApplication([])
    chat = NativeRichLogWidget()
    chat.set_logs([
        "You: Explain $x^2$",
        "Orthos: # Result\n\n**bold** and *italic*\n\n```python\nprint('$x^2$')\n```\n\n$$x^2+y^2=z^2$$\n\n$a_1 \\times b^2$",
    ])
    plain = chat.toPlainText()
    html = chat.document().toHtml()
    assert "Result" in plain
    assert "bold" in plain
    assert "print('$x^2$')" in plain  # Math inside code stays code.
    assert "<img" in html             # Actual equations are native images.


def test_native_math_images_are_transparent_not_white_rectangles():
    app = QApplication.instance() or QApplication([])
    chat = NativeRichLogWidget()
    chat._math_image(r"2^4 \times 3", False)
    # Resource URLs are sequential: the first formula is /f/1.
    image = chat.document().resource(
        QTextDocument.ResourceType.ImageResource, QUrl("orthos-math:/f/1")
    )
    assert image is not None, "math image was not registered as a resource"
    assert image.pixelColor(0, 0).alpha() == 0
    assert any(
        image.pixelColor(x, y).alpha() > 0
        for y in range(image.height()) for x in range(image.width())
    )


def test_role_cards_keep_user_ai_system_and_tool_visually_distinct():
    app = QApplication.instance() or QApplication([])
    chat = NativeRichLogWidget()
    chat.set_logs([
        "You: Hello", "Orthos: Hi", "SYS: Ready", "SYS: ▶ search_tools",
    ])
    html = chat.document().toHtml()
    assert NativeRichLogWidget._parse_entry("SYS: ▶ search_tools")["tag"] == "tool"
    # 2026 flat design: roles stay distinct via bubble tint, label color,
    # and row accent — verified through the shipped card builders.
    cards = [chat._card_html(m) for m in chat._messages]
    assert "background-color:#0d2f42" in cards[0]          # YOU bubble tint
    assert "#4dd8ff" in cards[1]                           # AI label accent
    assert "#d6a019" in cards[2]                           # SYS accent
    assert "#00a877" in cards[3]                           # TOOL accent


def test_activity_panel_uses_a_persisted_resizable_splitter():
    source = (os.path.dirname(os.path.dirname(__file__)) + "/ui.py")
    text = open(source, encoding="utf-8").read()
    assert "self._chat_splitter = QSplitter(Qt.Orientation.Horizontal)" in text
    assert "self._chat_splitter.splitterMoved.connect(self._save_activity_panel_width)" in text
    assert "layout/activity_panel_width" in text
