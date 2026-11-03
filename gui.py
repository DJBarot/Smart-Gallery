

import hashlib
import shutil
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import cv2
import numpy as np
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageDraw

from src.database import get_connection
from src.similarity import find_similar_images
from src.text_search import (
    search_by_text,
    search_by_image_clip
)

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_FOLDER = PROJECT_ROOT / "images"
IMAGE_MANAGER = PROJECT_ROOT / "src" / "image_manager.py"

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

# ═══════════════════════════════════════════════════════
# DESIGN TOKENS — Light Theme
# ═══════════════════════════════════════════════════════
BG_MAIN = "#F8F9FA"
BG_SIDEBAR = "#FFFFFF"
BG_CARD = "#FFFFFF"
ACCENT = "#268C83"
ACCENT_LIGHT = "#E8F3F1"
ACCENT_HOVER = "#1F756E"
TEXT_PRIMARY = "#252A34"
TEXT_SECONDARY = "#737B87"
BORDER = "#E8EAED"
HOVER_BG = "#F1F5F4"
DANGER = "#D94F57"
DANGER_HOVER = "#C43E46"
SUCCESS = "#268C83"
WHITE = "#FFFFFF"
SIDEBAR_WIDTH = 240
CARD_RADIUS = 12
FONT_FAMILY = "Segoe UI"

# Nav items definition
NAV_ITEMS = [
    ("gallery", "🖼", "All Images"),
    ("text_search", "🔍", "Search by Text"),
    ("image_search", "📷", "Search by Image"),
    ("processing", "⚙", "Image Processing"),
    ("import", "📥", "Import Images"),
    ("index", "⚡", "Index Images"),
]


class SmartImageGallery:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Image Gallery")
        self.root.geometry("1280x820")
        self.root.minsize(960, 640)
        self.root.configure(bg=BG_MAIN)

        self.events = queue.Queue()
        self.gallery = []
        self.gallery_page = 0
        self.gallery_page_size = 60
        self.thumbnail_refs = []
        self.search_image_path = None
        self.busy = False
        self.active_nav = "gallery"
        self.nav_buttons = {}
        self.toast_after_id = None

        self.setup_styles()
        self.build_interface()
        self.root.after(100, self.process_events)

    # ═══════════════════════════════════════════════════
    # STYLES
    # ═══════════════════════════════════════════════════
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Combobox styling
        style.configure(
            "TCombobox",
            fieldbackground=WHITE,
            background=WHITE,
            foreground=TEXT_PRIMARY,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
            arrowcolor=TEXT_SECONDARY,
            selectbackground=ACCENT_LIGHT,
            selectforeground=TEXT_PRIMARY,
        )
        style.map("TCombobox",
                   fieldbackground=[("readonly", WHITE)],
                   selectbackground=[("readonly", ACCENT_LIGHT)],
                   selectforeground=[("readonly", TEXT_PRIMARY)],
                   bordercolor=[("focus", ACCENT)],
                   )

        # Scrollbar
        style.configure(
            "TScrollbar",
            background=BG_MAIN,
            troughcolor=BG_MAIN,
            bordercolor=BG_MAIN,
            arrowcolor=TEXT_SECONDARY,
            lightcolor=BG_MAIN,
            darkcolor=BG_MAIN,
        )
        style.map("TScrollbar",
                   background=[("active", BORDER)],
                   )

        # Custom progress bar
        style.configure(
            "Accent.Horizontal.TProgressbar",
            troughcolor=ACCENT_LIGHT,
            background=ACCENT,
            bordercolor=ACCENT_LIGHT,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
        )

    # ═══════════════════════════════════════════════════
    # MAIN LAYOUT
    # ═══════════════════════════════════════════════════
    def build_interface(self):
        # ── Sidebar ──
        self.sidebar = tk.Frame(self.root, bg=BG_SIDEBAR, width=SIDEBAR_WIDTH)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Sidebar top border accent
        tk.Frame(self.sidebar, bg=ACCENT, height=3).pack(fill="x")

        # Logo area
        logo_frame = tk.Frame(self.sidebar, bg=BG_SIDEBAR)
        logo_frame.pack(fill="x", padx=20, pady=(20, 8))

        tk.Label(
            logo_frame,
            text="📸",
            bg=BG_SIDEBAR,
            fg=ACCENT,
            font=(FONT_FAMILY, 22),
        ).pack(side="left", padx=(0, 8))

        logo_text_frame = tk.Frame(logo_frame, bg=BG_SIDEBAR)
        logo_text_frame.pack(side="left")
        tk.Label(
            logo_text_frame,
            text="SmartImage",
            bg=BG_SIDEBAR,
            fg=TEXT_PRIMARY,
            font=(FONT_FAMILY, 14, "bold"),
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            logo_text_frame,
            text="Gallery",
            bg=BG_SIDEBAR,
            fg=TEXT_SECONDARY,
            font=(FONT_FAMILY, 11),
            anchor="w",
        ).pack(anchor="w")

        # Divider
        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(12, 16))

        # Navigation section label
        tk.Label(
            self.sidebar,
            text="NAVIGATION",
            bg=BG_SIDEBAR,
            fg=TEXT_SECONDARY,
            font=(FONT_FAMILY, 8, "bold"),
            anchor="w",
        ).pack(fill="x", padx=24, pady=(0, 6))

        # Navigation items
        nav_commands = {
            "gallery": self.show_gallery,
            "text_search": self.show_text_search,
            "image_search": self.show_image_search,
            "processing": self.show_image_processing,
            "import": self.import_images_from_folder,
            "index": self.index_images,
        }

        for key, icon, label in NAV_ITEMS:
            self.add_nav_button(self.sidebar, key, icon, label, nav_commands[key])

        # Sidebar bottom info
        bottom_frame = tk.Frame(self.sidebar, bg=BG_SIDEBAR)
        bottom_frame.pack(side="bottom", fill="x", padx=16, pady=16)

        tk.Frame(bottom_frame, bg=BORDER, height=1).pack(fill="x", pady=(0, 12))

        tk.Label(
            bottom_frame,
            text="Oracle Database",
            bg=BG_SIDEBAR,
            fg=TEXT_SECONDARY,
            font=(FONT_FAMILY, 8, "bold"),
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            bottom_frame,
            text="OpenCV • CLIP",
            bg=BG_SIDEBAR,
            fg=TEXT_SECONDARY,
            font=(FONT_FAMILY, 8),
            anchor="w",
        ).pack(anchor="w")

        # ── Sidebar right border ──
        tk.Frame(self.root, bg=BORDER, width=1).pack(side="left", fill="y")

        # ── Main Content Area ──
        main = tk.Frame(self.root, bg=BG_MAIN)
        main.pack(side="left", fill="both", expand=True)

        # Header bar
        header = tk.Frame(main, bg=BG_MAIN)
        header.pack(fill="x", padx=32, pady=(24, 0))

        title_frame = tk.Frame(header, bg=BG_MAIN)
        title_frame.pack(side="left", fill="x", expand=True)

        self.page_title = tk.Label(
            title_frame,
            text="Overview",
            bg=BG_MAIN,
            fg=TEXT_PRIMARY,
            font=(FONT_FAMILY, 22, "bold"),
            anchor="w",
        )
        self.page_title.pack(anchor="w")

        self.page_subtitle = tk.Label(
            title_frame,
            text="Welcome to your image gallery",
            bg=BG_MAIN,
            fg=TEXT_SECONDARY,
            font=(FONT_FAMILY, 10),
            anchor="w",
        )
        self.page_subtitle.pack(anchor="w", pady=(2, 0))

        self.count_label = tk.Label(
            header,
            text="",
            bg=BG_MAIN,
            fg=TEXT_SECONDARY,
            font=(FONT_FAMILY, 10),
        )
        self.count_label.pack(side="right", pady=10)

        # Status bar
        status_frame = tk.Frame(main, bg=BG_MAIN)
        status_frame.pack(fill="x", padx=32, pady=(8, 4))

        self.status = tk.Label(
            status_frame,
            text="Ready",
            bg=BG_MAIN,
            fg=ACCENT,
            anchor="w",
            font=(FONT_FAMILY, 9),
        )
        self.status.pack(side="left")

        # Toast notification area (hidden by default)
        self.toast_frame = tk.Frame(main, bg=BG_MAIN)
        self.toast_frame.pack(fill="x", padx=32, pady=(0, 0))
        self.toast_label = tk.Label(
            self.toast_frame,
            text="",
            bg=BG_MAIN,
            fg=BG_MAIN,
            font=(FONT_FAMILY, 9),
            pady=0,
        )

        # Content area
        self.content = tk.Frame(main, bg=BG_MAIN)
        self.content.pack(fill="both", expand=True, padx=24, pady=(4, 20))

        self.show_gallery()

    # ═══════════════════════════════════════════════════
    # TOAST NOTIFICATIONS
    # ═══════════════════════════════════════════════════
    def show_toast(self, message, toast_type="success"):
        """Show a subtle toast notification that fades after 3 seconds."""
        if self.toast_after_id:
            self.root.after_cancel(self.toast_after_id)

        colors = {
            "success": (ACCENT_LIGHT, ACCENT),
            "error": ("#FDE8E8", DANGER),
            "info": (ACCENT_LIGHT, TEXT_PRIMARY),
        }
        bg, fg = colors.get(toast_type, colors["info"])

        self.toast_label.config(text=f"  {message}  ", bg=bg, fg=fg)
        self.toast_label.pack(fill="x", ipady=6)

        self.toast_after_id = self.root.after(3500, self.hide_toast)

    def hide_toast(self):
        self.toast_label.pack_forget()
        self.toast_after_id = None

    # ═══════════════════════════════════════════════════
    # NAVIGATION
    # ═══════════════════════════════════════════════════
    def add_nav_button(self, parent, key, icon, label, command):
        btn_frame = tk.Frame(parent, bg=BG_SIDEBAR, cursor="hand2")
        btn_frame.pack(fill="x", padx=12, pady=2)

        is_active = (key == self.active_nav)
        bg = ACCENT_LIGHT if is_active else BG_SIDEBAR
        fg = ACCENT if is_active else TEXT_SECONDARY
        font_weight = "bold" if is_active else "normal"

        btn_frame.configure(bg=bg)

        icon_label = tk.Label(
            btn_frame,
            text=icon,
            bg=bg,
            fg=fg,
            font=(FONT_FAMILY, 11),
            cursor="hand2",
        )
        icon_label.pack(side="left", padx=(12, 8), pady=9)

        text_label = tk.Label(
            btn_frame,
            text=label,
            bg=bg,
            fg=fg,
            font=(FONT_FAMILY, 10, font_weight),
            anchor="w",
            cursor="hand2",
        )
        text_label.pack(side="left", fill="x", expand=True, pady=9)

        # Store references for updating active state
        self.nav_buttons[key] = (btn_frame, icon_label, text_label)

        def on_click(event=None):
            self.set_active_nav(key)
            command()

        def on_enter(event=None):
            if key != self.active_nav:
                btn_frame.configure(bg=HOVER_BG)
                icon_label.configure(bg=HOVER_BG)
                text_label.configure(bg=HOVER_BG)

        def on_leave(event=None):
            if key != self.active_nav:
                btn_frame.configure(bg=BG_SIDEBAR)
                icon_label.configure(bg=BG_SIDEBAR)
                text_label.configure(bg=BG_SIDEBAR)

        for widget in (btn_frame, icon_label, text_label):
            widget.bind("<Button-1>", on_click)
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)

    def set_active_nav(self, key):
        """Update the visual state of navigation buttons."""
        self.active_nav = key
        for nav_key, (frame, icon_lbl, text_lbl) in self.nav_buttons.items():
            if nav_key == key:
                frame.configure(bg=ACCENT_LIGHT)
                icon_lbl.configure(bg=ACCENT_LIGHT, fg=ACCENT)
                text_lbl.configure(bg=ACCENT_LIGHT, fg=ACCENT, font=(FONT_FAMILY, 10, "bold"))
            else:
                frame.configure(bg=BG_SIDEBAR)
                icon_lbl.configure(bg=BG_SIDEBAR, fg=TEXT_SECONDARY)
                text_lbl.configure(bg=BG_SIDEBAR, fg=TEXT_SECONDARY, font=(FONT_FAMILY, 10, "normal"))

    # ═══════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════
    def clear_content(self):
        for widget in self.content.winfo_children():
            widget.destroy()
        self.thumbnail_refs = []

    def set_page_title(self, title, subtitle=""):
        self.page_title.config(text=title)
        self.page_subtitle.config(text=subtitle)

    def set_status(self, message):
        self.status.config(text=message)

    def make_button(self, parent, text, command, style="primary", icon=None, width=None):
        """Create a styled button matching the design system."""
        styles = {
            "primary": (ACCENT, WHITE, ACCENT_HOVER),
            "secondary": (WHITE, TEXT_PRIMARY, HOVER_BG),
            "danger": (DANGER, WHITE, DANGER_HOVER),
            "ghost": (BG_MAIN, TEXT_SECONDARY, HOVER_BG),
        }
        bg, fg, hover = styles.get(style, styles["primary"])

        display_text = f"{icon}  {text}" if icon else text

        btn = tk.Button(
            parent,
            text=display_text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=hover,
            activeforeground=fg,
            relief="flat",
            bd=0,
            padx=16,
            pady=9,
            font=(FONT_FAMILY, 10),
            cursor="hand2",
            highlightthickness=0,
            width=width,
        )

        if style == "secondary":
            btn.configure(
                highlightbackground=BORDER,
                highlightthickness=1,
            )

        def on_enter(e):
            btn.configure(bg=hover)

        def on_leave(e):
            btn.configure(bg=bg)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

        return btn

    def make_card(self, parent, **kwargs):
        """Create a card-style frame."""
        card = tk.Frame(
            parent,
            bg=BG_CARD,
            padx=kwargs.get("padx", 16),
            pady=kwargs.get("pady", 16),
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        return card

    # ═══════════════════════════════════════════════════
    # DATABASE HELPERS (unchanged logic)
    # ═══════════════════════════════════════════════════
    def get_gallery(self):
        connection = get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute("""
                SELECT image_id, filename, file_path
                FROM images
                ORDER BY image_id
            """)
            return cursor.fetchall()
        finally:
            cursor.close()
            connection.close()

    def refresh_gallery(self):
        try:
            self.gallery = self.get_gallery()
            self.count_label.config(
                text=f"{len(self.gallery)} indexed images"
            )
            self.set_status("Gallery refreshed.")
            return True
        except Exception as error:
            self.set_status("Database connection failed.")
            messagebox.showerror(
                "Database Error",
                f"Could not load gallery:\n{error}"
            )
            return False

    # ═══════════════════════════════════════════════════
    # ALL IMAGES (GALLERY) PAGE
    # ═══════════════════════════════════════════════════
    def show_gallery(self):
        """Refresh Oracle records and display the first gallery page."""
        self.set_active_nav("gallery")
        if not self.refresh_gallery():
            return
        self.gallery_page = 0
        self.render_gallery_page()

    def render_gallery_page(self):
        """Render only the current page to keep Tkinter responsive."""
        self.clear_content()
        self.set_page_title("All Images", "Browse your complete image collection")

        total_images = len(self.gallery)
        page_size = self.gallery_page_size
        total_pages = max(1, (total_images + page_size - 1) // page_size)
        self.gallery_page = max(0, min(self.gallery_page, total_pages - 1))

        start = self.gallery_page * page_size
        end = min(start + page_size, total_images)
        page_records = self.gallery[start:end]

        self.count_label.config(text=f"{total_images} indexed images")

        # Toolbar
        toolbar = tk.Frame(self.content, bg=BG_MAIN)
        toolbar.pack(fill="x", pady=(0, 12))

        tk.Label(
            toolbar,
            text=f"Showing {start + 1 if total_images else 0}–{end} of {total_images} images",
            bg=BG_MAIN,
            fg=TEXT_SECONDARY,
            font=(FONT_FAMILY, 10),
        ).pack(side="left")

        self.make_button(
            toolbar, "Refresh", self.show_gallery, "secondary", "🔄"
        ).pack(side="right")

        self.render_cards(page_records, show_scores=False)

        # Pagination
        pagination = tk.Frame(self.content, bg=BG_MAIN)
        pagination.pack(fill="x", pady=(10, 0))

        prev_btn = self.make_button(
            pagination, "← Previous", self.previous_gallery_page,
            "secondary" if self.gallery_page > 0 else "ghost"
        )
        if self.gallery_page <= 0:
            prev_btn.configure(state="disabled", fg=TEXT_SECONDARY)
        prev_btn.pack(side="left")

        tk.Label(
            pagination,
            text=f"Page {self.gallery_page + 1} of {total_pages}",
            bg=BG_MAIN,
            fg=TEXT_SECONDARY,
            font=(FONT_FAMILY, 10, "bold"),
        ).pack(side="left", expand=True)

        next_btn = self.make_button(
            pagination, "Next →", self.next_gallery_page,
            "secondary" if self.gallery_page < total_pages - 1 else "ghost"
        )
        if self.gallery_page >= total_pages - 1:
            next_btn.configure(state="disabled", fg=TEXT_SECONDARY)
        next_btn.pack(side="right")

    def previous_gallery_page(self):
        if self.gallery_page > 0:
            self.gallery_page -= 1
            self.render_gallery_page()

    def next_gallery_page(self):
        total_pages = max(
            1,
            (len(self.gallery) + self.gallery_page_size - 1)
            // self.gallery_page_size
        )
        if self.gallery_page < total_pages - 1:
            self.gallery_page += 1
            self.render_gallery_page()

    # ═══════════════════════════════════════════════════
    # IMAGE CARDS RENDERER
    # ═══════════════════════════════════════════════════
    def resolve_image_path(self, file_path, filename=None):
        """Resolve an image path from the database, including moved project folders."""
        candidates = []
        if file_path:
            stored = Path(str(file_path)).expanduser()
            candidates.append(stored)
            if not stored.is_absolute():
                candidates.append(PROJECT_ROOT / stored)
                candidates.append(IMAGE_FOLDER / stored)
        if filename:
            candidates.append(IMAGE_FOLDER / Path(str(filename)).name)

        for candidate in candidates:
            try:
                if candidate.is_file():
                    return str(candidate.resolve())
            except OSError:
                pass

        # Database paths may point to the original machine or an older project
        # location. Recover by filename from the project's image directory.
        if filename and IMAGE_FOLDER.exists():
            target_name = Path(str(filename)).name.lower()
            try:
                for candidate in IMAGE_FOLDER.rglob('*'):
                    if candidate.is_file() and candidate.name.lower() == target_name:
                        return str(candidate.resolve())
            except OSError:
                pass
        return str(file_path) if file_path else ""

    def render_cards(self, records, show_scores=False, parent=None, scrollable=True):
        """Render image cards. Set scrollable=False when the parent already scrolls."""
        records = [
            {
                "image_id": record[0],
                "filename": record[1],
                "file_path": record[2]
            }
            if isinstance(record, (tuple, list))
            else dict(record)
            for record in records
        ]
        for record in records:
            record["file_path"] = self.resolve_image_path(
                record.get("file_path"), record.get("filename")
            )

        host = parent if parent is not None else self.content

        if scrollable:
            container = tk.Frame(host, bg=BG_MAIN)
            container.pack(fill="both", expand=True)

            canvas = tk.Canvas(
                container,
                bg=BG_MAIN,
                highlightthickness=0,
                borderwidth=0
            )
            scrollbar = ttk.Scrollbar(
                container,
                orient="vertical",
                command=canvas.yview
            )
            inner = tk.Frame(canvas, bg=BG_MAIN)

            inner.bind(
                "<Configure>",
                lambda event: canvas.configure(
                    scrollregion=canvas.bbox("all")
                )
            )
            canvas_window = canvas.create_window(
                (0, 0), window=inner, anchor="nw"
            )
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.bind(
                "<Configure>",
                lambda event: canvas.itemconfigure(
                    canvas_window, width=event.width
                )
            )

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
        else:
            inner = host
            for child in inner.winfo_children():
                child.destroy()

        if not records:
            tk.Label(
                inner,
                text="No images to display.",
                bg=BG_MAIN,
                fg=TEXT_SECONDARY,
                font=(FONT_FAMILY, 12),
            ).grid(row=0, column=0, sticky="w", padx=10, pady=18)
            return

        columns = 3
        for column in range(columns):
            inner.grid_columnconfigure(column, weight=1, uniform="image_cards")

        for index, record in enumerate(records):
            row, column = divmod(index, columns)
            card = tk.Frame(
                inner,
                bg=BG_CARD,
                padx=12,
                pady=12,
                highlightbackground=BORDER,
                highlightthickness=1,
            )
            card.grid(
                row=row,
                column=column,
                padx=6,
                pady=6,
                sticky="nsew"
            )
            card.bind(
                "<Button-1>",
                lambda event, idx=index: self.open_image_viewer(records, idx)
            )
            card.configure(cursor="hand2")

            # Hover effects
            def on_enter(e, c=card):
                c.configure(highlightbackground=ACCENT, highlightthickness=2)

            def on_leave(e, c=card):
                c.configure(highlightbackground=BORDER, highlightthickness=1)

            card.bind("<Enter>", on_enter)
            card.bind("<Leave>", on_leave)

            try:
                with Image.open(record["file_path"]) as source:
                    thumbnail = source.convert("RGB")
                    thumbnail.thumbnail((210, 140))
                    photo = ImageTk.PhotoImage(thumbnail)

                self.thumbnail_refs.append(photo)
                image_label = tk.Label(
                    card,
                    image=photo,
                    bg=BG_CARD,
                    cursor="hand2"
                )
                image_label.pack(pady=(0, 8))
                image_label.bind(
                    "<Button-1>",
                    lambda event, idx=index: self.open_image_viewer(records, idx)
                )
            except (OSError, ValueError, KeyError):
                tk.Label(
                    card,
                    text="⚠ Image unavailable",
                    bg=BG_CARD,
                    fg=DANGER,
                    width=24,
                    height=6,
                    font=(FONT_FAMILY, 9),
                ).pack()

            tk.Label(
                card,
                text=record.get("filename", "Unknown filename"),
                bg=BG_CARD,
                fg=TEXT_PRIMARY,
                wraplength=210,
                justify="left",
                font=(FONT_FAMILY, 9, "bold"),
            ).pack(anchor="w")

            tk.Label(
                card,
                text=f"ID: {record.get('image_id', '—')}",
                bg=BG_CARD,
                fg=TEXT_SECONDARY,
                font=(FONT_FAMILY, 8),
            ).pack(anchor="w", pady=(4, 0))

            if show_scores and "similarity_score" in record:
                score = record["similarity_score"]
                score_text = f"CLIP similarity: {score:.4f}"
                score_color = ACCENT if score > 0.3 else TEXT_SECONDARY
                tk.Label(
                    card,
                    text=score_text,
                    bg=BG_CARD,
                    fg=score_color,
                    font=(FONT_FAMILY, 8, "bold"),
                ).pack(anchor="w", pady=(5, 0))
            elif show_scores and "similarity_distance" in record:
                dist = record["similarity_distance"]
                tk.Label(
                    card,
                    text=f"Distance: {dist:.4f}",
                    bg=BG_CARD,
                    fg=ACCENT,
                    font=(FONT_FAMILY, 8, "bold"),
                ).pack(anchor="w", pady=(5, 0))

            for widget in (card, *card.winfo_children()):
                widget.configure(cursor="hand2")
                widget.bind(
                    "<Button-1>",
                    lambda event, items=records, selected=index: self.open_image_viewer(items, selected)
                )

    # ═══════════════════════════════════════════════════
    # IMAGE VIEWER
    # ═══════════════════════════════════════════════════
    def open_image_viewer(self, records, index=0):
        """Full-resolution viewer with centered fit, mouse-wheel scrolling and safe removal."""

        normalized = []
        for record in records:
            if isinstance(record, (tuple, list)):
                normalized.append({
                    "image_id": record[0], "filename": record[1], "file_path": record[2]
                })
            else:
                normalized.append(dict(record))

        for record in normalized:
            record["file_path"] = self.resolve_image_path(
                record.get("file_path"), record.get("filename")
            )
        normalized = [r for r in normalized if r.get("file_path") and Path(r["file_path"]).is_file()]
        if not normalized:
            messagebox.showwarning("Image unavailable", "There are no image files to view.")
            return

        index = max(0, min(index, len(normalized) - 1))
        window = tk.Toplevel(self.root)
        window.title("Smart Image Gallery • Image Viewer")
        window.geometry("1280x860")
        window.minsize(760, 540)
        window.configure(bg=BG_MAIN)

        # Top toolbar
        toolbar = tk.Frame(window, bg=WHITE, padx=14, pady=10)
        toolbar.pack(fill="x")
        tk.Frame(window, bg=BORDER, height=1).pack(fill="x")

        info = tk.Label(toolbar, text="", bg=WHITE, fg=TEXT_PRIMARY,
                        anchor="w", font=(FONT_FAMILY, 11, "bold"))
        info.pack(side="left", fill="x", expand=True, padx=(0, 10))

        # Canvas
        canvas_frame = tk.Frame(window, bg=BG_MAIN)
        canvas_frame.pack(fill="both", expand=True, padx=12, pady=(8, 5))
        image_canvas = tk.Canvas(canvas_frame, bg="#F0F2F5", highlightthickness=0,
                                 borderwidth=0, takefocus=True)
        xbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=image_canvas.xview)
        ybar = ttk.Scrollbar(canvas_frame, orient="vertical", command=image_canvas.yview)
        image_canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        image_canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        # Details bar
        details_bar = tk.Frame(window, bg=WHITE, pady=8, padx=14)
        details_bar.pack(fill="x", padx=12, pady=(4, 10))
        details = tk.Label(details_bar, text="", bg=WHITE, fg=TEXT_SECONDARY,
                           justify="left", anchor="w",
                           wraplength=1100, font=(FONT_FAMILY, 9))
        details.pack(fill="x")

        state = {"index": index, "image": None, "photo": None,
                 "scale": 0, "fit_scale": 1.0, "item": None}

        def draw_image(reset_fit=False):
            current = normalized[state["index"]]
            path = current.get("file_path", "")
            try:
                with Image.open(path) as source:
                    original = source.copy()
                    image_format = source.format or Path(path).suffix.upper().lstrip(".")
            except (OSError, ValueError) as error:
                messagebox.showerror("Unable to open image",
                                     f"Could not open image:\n{path}\n\n{error}",
                                     parent=window)
                return

            state["image"] = original
            width, height = original.size
            try:
                file_size = os.path.getsize(path)
                size_text = (f"{file_size / (1024 * 1024):.2f} MB"
                             if file_size >= 1024 * 1024 else f"{file_size / 1024:.1f} KB")
            except OSError:
                size_text = "Unknown"

            info.config(text=f"{state['index'] + 1} / {len(normalized)}   •   {current.get('filename', os.path.basename(path))}")
            details.config(text=(
                f"Image ID: {current.get('image_id', '—')}   |   Dimensions: {width} × {height} px   |   "
                f"Format: {image_format}   |   File size: {size_text}\n{path}"
            ))

            window.update_idletasks()
            viewport_w = max(200, image_canvas.winfo_width())
            viewport_h = max(150, image_canvas.winfo_height())
            if reset_fit or state["scale"] <= 0:
                state["fit_scale"] = min(viewport_w / width, viewport_h / height, 1.0)
                state["scale"] = state["fit_scale"]

            display_w = max(1, int(width * state["scale"]))
            display_h = max(1, int(height * state["scale"]))
            shown = (original.resize((display_w, display_h), Image.Resampling.LANCZOS)
                     if (display_w, display_h) != original.size else original)
            state["photo"] = ImageTk.PhotoImage(shown)
            image_canvas.delete("all")
            canvas_w = max(viewport_w, display_w)
            canvas_h = max(viewport_h, display_h)
            x = max(0, (viewport_w - display_w) // 2)
            y = max(0, (viewport_h - display_h) // 2)
            state["item"] = image_canvas.create_image(x, y, anchor="nw", image=state["photo"])
            image_canvas.configure(scrollregion=(0, 0, canvas_w, canvas_h))
            image_canvas.xview_moveto(0)
            image_canvas.yview_moveto(0)

        def navigate(delta):
            next_index = state["index"] + delta
            if 0 <= next_index < len(normalized):
                state["index"] = next_index
                state["scale"] = 0
                draw_image(reset_fit=True)

        def zoom(factor):
            if state["image"] is None:
                return
            state["scale"] = min(5.0, max(0.1, state["scale"] * factor))
            draw_image()

        def delete_from_gallery():
            current = normalized[state["index"]]
            filename = current.get("filename", os.path.basename(current["file_path"]))
            if not messagebox.askyesno(
                "Remove image from gallery",
                f"Remove '{filename}' from the gallery?\n\n"
                "The image file will be moved to a recoverable .gallery_trash folder "
                "and its Oracle index records will be removed.",
                parent=window
            ):
                return

            path = Path(current["file_path"])
            trash_dir = PROJECT_ROOT / ".gallery_trash"
            try:
                trash_dir.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    target = trash_dir / path.name
                    suffix_index = 1
                    while target.exists():
                        target = trash_dir / f"{path.stem}_{suffix_index}{path.suffix}"
                        suffix_index += 1
                    shutil.move(str(path), str(target))

                connection = get_connection()
                cursor = connection.cursor()
                try:
                    image_id = current.get("image_id")
                    cursor.execute("DELETE FROM image_features WHERE image_id = :image_id",
                                   {"image_id": image_id})
                    cursor.execute("DELETE FROM images WHERE image_id = :image_id",
                                   {"image_id": image_id})
                    connection.commit()
                except Exception:
                    connection.rollback()
                    if 'target' in locals() and target.exists() and not path.exists():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(target), str(path))
                    raise
                finally:
                    cursor.close()
                    connection.close()

                normalized.pop(state["index"])
                self.refresh_gallery()
                if not normalized:
                    window.destroy()
                    self.show_gallery()
                    messagebox.showinfo("Image removed", "The image was removed. The gallery is now empty.")
                    return
                state["index"] = min(state["index"], len(normalized) - 1)
                state["scale"] = 0
                draw_image(reset_fit=True)
                self.set_status("Image removed from gallery. File kept in .gallery_trash.")
            except Exception as error:
                try:
                    if 'target' in locals() and target.exists() and not path.exists():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(target), str(path))
                except OSError:
                    pass
                messagebox.showerror("Could not remove image", str(error), parent=window)

        def on_mousewheel(event):
            if getattr(event, "num", None) == 4:
                image_canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", None) == 5:
                image_canvas.yview_scroll(3, "units")
            elif event.delta:
                if abs(event.delta) >= 120:
                    image_canvas.yview_scroll(int(-event.delta / 120) * 3, "units")
                else:
                    image_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        def on_shift_mousewheel(event):
            if event.delta:
                image_canvas.xview_scroll(int(-event.delta / 120) * 3 or (-1 if event.delta > 0 else 1), "units")

        def bind_wheel(_event=None):
            image_canvas.bind_all("<MouseWheel>", on_mousewheel)
            image_canvas.bind_all("<Button-4>", on_mousewheel)
            image_canvas.bind_all("<Button-5>", on_mousewheel)
            image_canvas.bind_all("<Shift-MouseWheel>", on_shift_mousewheel)

        def unbind_wheel(_event=None):
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>", "<Shift-MouseWheel>"):
                window.unbind_all(sequence)

        image_canvas.bind("<Enter>", bind_wheel)
        image_canvas.bind("<Leave>", unbind_wheel)
        window.bind("<Destroy>", lambda event: unbind_wheel() if event.widget is window else None)

        # Controls
        controls = tk.Frame(toolbar, bg=WHITE)
        controls.pack(side="right")

        def make_viewer_button(parent, label, command, bg_color=WHITE, fg_color=TEXT_PRIMARY, width=None):
            button = tk.Button(
                parent,
                text=label,
                command=command,
                bg=bg_color,
                fg=fg_color,
                activebackground=HOVER_BG,
                activeforeground=fg_color,
                relief="flat",
                bd=0,
                padx=10,
                pady=7,
                cursor="hand2",
                font=(FONT_FAMILY, 9),
                width=width,
                highlightthickness=0,
            )
            button.pack(side="left", padx=(0, 4))

            def on_enter(e):
                button.configure(bg=HOVER_BG)

            def on_leave(e):
                button.configure(bg=bg_color)

            button.bind("<Enter>", on_enter)
            button.bind("<Leave>", on_leave)
            return button

        # Actions
        actions_group = tk.Frame(controls, bg=WHITE)
        actions_group.pack(side="left", padx=(0, 8))
        make_viewer_button(
            actions_group, "🗑 Remove", delete_from_gallery,
            WHITE, DANGER, width=10
        )

        # Zoom
        zoom_group = tk.Frame(controls, bg=WHITE)
        zoom_group.pack(side="left", padx=(0, 8))
        make_viewer_button(zoom_group, "−", lambda: zoom(1 / 1.25), width=3)
        make_viewer_button(zoom_group, "+", lambda: zoom(1.25), width=3)
        make_viewer_button(
            zoom_group, "Fit", lambda: draw_image(reset_fit=True),
            ACCENT_LIGHT, ACCENT, width=6
        )

        # Navigation
        nav_group = tk.Frame(controls, bg=WHITE)
        nav_group.pack(side="left")
        make_viewer_button(nav_group, "◀", lambda: navigate(-1), width=4)
        make_viewer_button(nav_group, "▶", lambda: navigate(1), width=4)

        window.bind("<Left>", lambda event: navigate(-1))
        window.bind("<Right>", lambda event: navigate(1))
        window.bind("<Control-plus>", lambda event: zoom(1.25))
        window.bind("<Control-minus>", lambda event: zoom(1 / 1.25))
        draw_image(reset_fit=True)

    # ═══════════════════════════════════════════════════
    # IMAGE PROCESSING PAGE
    # ═══════════════════════════════════════════════════
    def show_image_processing(self):
        """A standalone image-processing workspace; does not alter search/database code."""
        self.set_active_nav("processing")
        self.clear_content()
        self.set_page_title("Image Processing", "Apply operations to images — originals are never overwritten")

        self.processing_source_image = None
        self.processing_output_image = None
        self.processing_original_photo = None
        self.processing_result_photo = None

        # Controls card
        controls_card = self.make_card(self.content, padx=20, pady=16)
        controls_card.pack(fill="x", pady=(0, 16))

        self.processing_filename_label = tk.Label(
            controls_card,
            text="No image selected",
            bg=BG_CARD,
            fg=ACCENT,
            font=(FONT_FAMILY, 10, "bold"),
        )
        self.processing_filename_label.grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 12))

        def choose_processing_image():
            path = filedialog.askopenfilename(
                title="Choose image to process",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                    ("All files", "*.*")
                ]
            )
            if not path:
                return
            try:
                with Image.open(path) as image_file:
                    self.processing_source_image = image_file.convert("RGB").copy()
                self.processing_output_image = None
                self.processing_filename_label.config(text=f"📎 {os.path.basename(path)}")
                self._update_processing_preview(
                    self.processing_source_image,
                    self.processing_original_label,
                    "original"
                )
                self.processing_result_label.config(
                    image="", text="Choose an operation and click Apply"
                )
                self.processing_result_photo = None
                self.set_status("Image loaded for processing. Original file is unchanged.")
            except (OSError, ValueError) as error:
                messagebox.showerror("Unable to open image", str(error))

        self.make_button(
            controls_card, "Choose Image", choose_processing_image, "secondary", "📁"
        ).grid(row=1, column=0, sticky="w", padx=(0, 14))

        tk.Label(
            controls_card, text="Operation:", bg=BG_CARD, fg=TEXT_PRIMARY,
            font=(FONT_FAMILY, 9, "bold"),
        ).grid(row=1, column=1, sticky="w")

        self.processing_operation_var = tk.StringVar(value="Grayscale")
        ttk.Combobox(
            controls_card,
            textvariable=self.processing_operation_var,
            values=[
                "Grayscale",
                "Gaussian Blur",
                "Median Filter",
                "Canny Edge Detection",
                "Sharpen",
                "Brightness Adjustment",
                "Contrast Adjustment"
            ],
            state="readonly",
            width=22
        ).grid(row=1, column=2, sticky="w", padx=(8, 18))

        tk.Label(
            controls_card, text="Intensity:", bg=BG_CARD, fg=TEXT_PRIMARY,
            font=(FONT_FAMILY, 9, "bold"),
        ).grid(row=1, column=3, sticky="w")
        self.processing_intensity_var = tk.IntVar(value=3)
        tk.Spinbox(
            controls_card,
            from_=1,
            to=10,
            width=4,
            textvariable=self.processing_intensity_var,
            bg=WHITE,
            fg=TEXT_PRIMARY,
            highlightbackground=BORDER,
            highlightthickness=1,
            relief="flat",
        ).grid(row=1, column=4, sticky="w", padx=(8, 18))

        def apply_operation():
            if self.processing_source_image is None:
                messagebox.showwarning("No image", "Choose an image first.")
                return
            try:
                intensity = int(self.processing_intensity_var.get())
                if not 1 <= intensity <= 10:
                    raise ValueError("Intensity must be between 1 and 10.")
                source = self.processing_source_image.copy()
                operation = self.processing_operation_var.get()

                if operation == "Grayscale":
                    output = source.convert("L").convert("RGB")
                elif operation == "Gaussian Blur":
                    output = source.filter(ImageFilter.GaussianBlur(radius=intensity))
                elif operation == "Median Filter":
                    output = source.filter(ImageFilter.MedianFilter(size=2 * intensity + 1))
                elif operation == "Canny Edge Detection":
                    gray = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
                    low_threshold = max(15, 70 - (intensity * 5))
                    high_threshold = min(255, low_threshold * 2 + 50)
                    edges = cv2.Canny(
                        blurred,
                        low_threshold,
                        high_threshold,
                        L2gradient=True
                    )
                    output = Image.fromarray(edges).convert("RGB")
                elif operation == "Sharpen":
                    output = ImageEnhance.Sharpness(source).enhance(1.0 + intensity * 0.5)
                elif operation == "Brightness Adjustment":
                    output = ImageEnhance.Brightness(source).enhance(intensity / 5.0)
                elif operation == "Contrast Adjustment":
                    output = ImageEnhance.Contrast(source).enhance(intensity / 5.0)
                else:
                    raise ValueError("Select a supported operation.")

                self.processing_output_image = output
                self._update_processing_preview(
                    output, self.processing_result_label, "result"
                )
                self.set_status(f"Applied {operation}. Original file remains unchanged.")
                self.show_toast(f"✓ {operation} applied successfully", "success")
            except (ValueError, OSError, cv2.error) as error:
                messagebox.showerror("Processing failed", str(error))

        self.make_button(
            controls_card, "Apply Operation", apply_operation, "primary", "▶"
        ).grid(row=1, column=5, sticky="w")

        # Preview area
        preview_row = tk.Frame(self.content, bg=BG_MAIN)
        preview_row.pack(fill="both", expand=True)
        preview_row.grid_columnconfigure(0, weight=1, uniform="processing_preview")
        preview_row.grid_columnconfigure(1, weight=1, uniform="processing_preview")
        preview_row.grid_rowconfigure(1, weight=1)

        tk.Label(
            preview_row, text="Original Image", bg=BG_MAIN, fg=TEXT_PRIMARY,
            font=(FONT_FAMILY, 13, "bold")
        ).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
        tk.Label(
            preview_row, text="Processed Image", bg=BG_MAIN, fg=TEXT_PRIMARY,
            font=(FONT_FAMILY, 13, "bold")
        ).grid(row=0, column=1, sticky="w", padx=(8, 0), pady=(0, 8))

        self.processing_original_label = tk.Label(
            preview_row, text="Choose an image to begin", bg=BG_CARD, fg=TEXT_SECONDARY,
            highlightbackground=BORDER, highlightthickness=1,
        )
        self.processing_original_label.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 10))
        self.processing_result_label = tk.Label(
            preview_row, text="Processed output will appear here", bg=BG_CARD, fg=TEXT_SECONDARY,
            highlightbackground=BORDER, highlightthickness=1,
        )
        self.processing_result_label.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(0, 10))

        self.make_button(
            self.content, "Save Processed Image", self.save_processed_image, "secondary", "💾"
        ).pack(anchor="w", pady=(8, 0))

    def _update_processing_preview(self, image, target_label, which):
        """Resize a copy for display while retaining the full-resolution result."""
        preview = image.copy()
        preview.thumbnail((460, 330))
        photo = ImageTk.PhotoImage(preview)
        target_label.config(image=photo, text="")
        if which == "original":
            self.processing_original_photo = photo
        else:
            self.processing_result_photo = photo

    def save_processed_image(self):
        if self.processing_output_image is None:
            messagebox.showwarning(
                "No processed image",
                "Choose an image and apply an operation before saving."
            )
            return
        path = filedialog.asksaveasfilename(
            title="Save processed image",
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("JPEG image", "*.jpg *.jpeg"),
                ("Bitmap image", "*.bmp"),
                ("TIFF image", "*.tif *.tiff")
            ]
        )
        if not path:
            return
        try:
            output = self.processing_output_image
            suffix = Path(path).suffix.lower()
            if suffix in {".jpg", ".jpeg"} and output.mode != "RGB":
                output = output.convert("RGB")
            output.save(path)
            self.set_status(f"Processed image saved: {path}")
            self.show_toast("✓ Image saved successfully", "success")
        except (OSError, ValueError) as error:
            messagebox.showerror("Save failed", str(error))

    # ═══════════════════════════════════════════════════
    # SEARCH BY TEXT PAGE
    # ═══════════════════════════════════════════════════
    def show_text_search(self):
        self.set_active_nav("text_search")
        self.clear_content()
        self.set_page_title("Search by Text", "Describe the image you want to find using natural language")

        # Search card
        search_card = self.make_card(self.content, padx=20, pady=20)
        search_card.pack(fill="x", pady=(0, 16))

        tk.Label(
            search_card,
            text="🔍  What are you looking for?",
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            font=(FONT_FAMILY, 13, "bold"),
        ).pack(anchor="w", pady=(0, 12))

        query_var = tk.StringVar()

        entry = tk.Entry(
            search_card,
            textvariable=query_var,
            bg="#F5F6F8",
            fg=TEXT_PRIMARY,
            insertbackground=ACCENT,
            relief="flat",
            font=(FONT_FAMILY, 12),
            highlightbackground=BORDER,
            highlightthickness=1,
            highlightcolor=ACCENT,
        )
        entry.pack(fill="x", ipady=12, pady=(0, 12))

        # Placeholder handling
        placeholder = "Describe the image you are looking for..."

        def on_focus_in(e):
            if entry.get() == placeholder:
                entry.delete(0, tk.END)
                entry.config(fg=TEXT_PRIMARY)

        def on_focus_out(e):
            if not entry.get():
                entry.insert(0, placeholder)
                entry.config(fg=TEXT_SECONDARY)

        entry.insert(0, placeholder)
        entry.config(fg=TEXT_SECONDARY)
        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)

        controls = tk.Frame(search_card, bg=BG_CARD)
        controls.pack(fill="x")

        tk.Label(
            controls,
            text="Results:",
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            font=(FONT_FAMILY, 9),
        ).pack(side="left")

        count_var = tk.StringVar(value="5")
        ttk.Combobox(
            controls,
            textvariable=count_var,
            values=["1", "3", "5", "10", "20"],
            state="readonly",
            width=6
        ).pack(side="left", padx=8)

        def search():
            query = query_var.get().strip()
            if query == placeholder:
                query = ""
            if not query:
                messagebox.showwarning(
                    "Missing query",
                    "Enter a text description first."
                )
                return

            self.start_task(
                "Searching gallery with CLIP...",
                lambda: search_by_text(
                    query, top_n=int(count_var.get())
                ),
                self.show_text_results
            )

        self.make_button(
            controls, "Search Gallery", search, "primary", "🔍"
        ).pack(side="right")

        self.results_area = tk.Frame(self.content, bg=BG_MAIN)
        self.results_area.pack(fill="both", expand=True, pady=(0, 0))

        entry.bind("<Return>", lambda event: search())

    def show_text_results(self, results):
        self.set_page_title("Search by Text", "Describe the image you want to find using natural language")

        if not hasattr(self, "results_area"):
            return

        for widget in self.results_area.winfo_children():
            widget.destroy()

        self.thumbnail_refs = []

        result_header = tk.Frame(self.results_area, bg=BG_MAIN)
        result_header.pack(fill="x", pady=(8, 12))

        tk.Label(
            result_header,
            text=f"Search Results — {len(results)} found",
            bg=BG_MAIN,
            fg=TEXT_PRIMARY,
            font=(FONT_FAMILY, 14, "bold"),
        ).pack(side="left")

        if not results:
            empty_card = self.make_card(self.results_area, padx=20, pady=24)
            empty_card.pack(fill="x")
            tk.Label(
                empty_card,
                text="No relevant images found. Try another description.",
                bg=BG_CARD,
                fg=TEXT_SECONDARY,
                font=(FONT_FAMILY, 11),
            ).pack()
            return

        self.show_toast(f"✓ Found {len(results)} matching images", "success")
        self.render_cards(
            results,
            show_scores=True,
            parent=self.results_area
        )

    # ═══════════════════════════════════════════════════
    # SEARCH BY IMAGE PAGE
    # ═══════════════════════════════════════════════════
    def show_image_search(self):
        self.set_active_nav("image_search")
        self.clear_content()
        self.set_page_title("Search by Image", "Upload an image to find visually or semantically similar ones")
        self.search_image_path = None
        self.query_preview_image = None

        # Query card
        query_card = self.make_card(self.content, padx=20, pady=16)
        query_card.pack(fill="x", pady=(0, 16))

        inner = tk.Frame(query_card, bg=BG_CARD)
        inner.pack(fill="x")

        # Left: Preview
        preview_column = tk.Frame(inner, bg=BG_CARD)
        preview_column.pack(side="left", anchor="nw", padx=(0, 24))

        selected_label = tk.Label(
            preview_column,
            text="No image selected",
            bg=BG_CARD,
            fg=ACCENT,
            font=(FONT_FAMILY, 9, "bold"),
        )
        selected_label.pack(anchor="w", pady=(0, 6))

        self.query_preview_label = tk.Label(
            preview_column,
            text="📷 Drop or select an image",
            bg="#F5F6F8",
            fg=TEXT_SECONDARY,
            width=28,
            height=7,
            highlightbackground=BORDER,
            highlightthickness=1,
            font=(FONT_FAMILY, 9),
        )
        self.query_preview_label.pack(anchor="w")

        # Right: Controls
        controls_column = tk.Frame(inner, bg=BG_CARD)
        controls_column.pack(side="left", fill="y", anchor="nw")

        tk.Label(
            controls_column,
            text="Similarity method",
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            font=(FONT_FAMILY, 9, "bold"),
        ).pack(anchor="w", pady=(0, 4))

        method_var = tk.StringVar(value="CLIP semantic similarity")
        ttk.Combobox(
            controls_column,
            textvariable=method_var,
            values=["CLIP semantic similarity", "HSV color similarity"],
            state="readonly",
            width=28
        ).pack(anchor="w", pady=(0, 12))

        tk.Label(
            controls_column,
            text="Number of results",
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            font=(FONT_FAMILY, 9, "bold"),
        ).pack(anchor="w", pady=(0, 4))

        count_var = tk.StringVar(value="5")
        ttk.Combobox(
            controls_column,
            textvariable=count_var,
            values=["1", "3", "5", "10", "20"],
            state="readonly",
            width=8
        ).pack(anchor="w", pady=(0, 12))

        def choose_image():
            path = filedialog.askopenfilename(
                title="Choose query image",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                    ("All files", "*.*")
                ]
            )
            if not path:
                return

            try:
                with Image.open(path) as source:
                    preview = source.convert("RGB")
                    preview.thumbnail((220, 145))
                    preview = preview.copy()

                self.search_image_path = path
                self.query_preview_image = ImageTk.PhotoImage(preview)
                selected_label.config(text=f"📎 {os.path.basename(path)}")
                self.query_preview_label.config(
                    image=self.query_preview_image,
                    text="",
                    width=0,
                    height=0
                )
            except (OSError, ValueError) as error:
                messagebox.showerror("Unable to open image", str(error))

        btn_row = tk.Frame(controls_column, bg=BG_CARD)
        btn_row.pack(anchor="w", fill="x")

        self.make_button(
            btn_row, "Choose Image", choose_image, "secondary", "📁"
        ).pack(side="left", padx=(0, 8))

        def search():
            if not self.search_image_path:
                messagebox.showwarning("No image", "Choose a query image first.")
                return

            query_path = self.search_image_path
            top_n = int(count_var.get())

            if method_var.get() == "CLIP semantic similarity":
                search_function = lambda: search_by_image_clip(query_path, top_n=top_n)
            else:
                search_function = lambda: find_similar_images(query_path, top_n=top_n)

            self.start_task(
                "Searching for similar images...",
                search_function,
                self.show_image_search_results
            )

        self.make_button(
            btn_row, "Find Similar", search, "primary", "🔍"
        ).pack(side="left")

        # Results area
        tk.Label(
            self.content,
            text="Similar Images",
            bg=BG_MAIN,
            fg=TEXT_PRIMARY,
            font=(FONT_FAMILY, 14, "bold"),
        ).pack(anchor="w", pady=(4, 8))

        results_outer = tk.Frame(self.content, bg=BG_MAIN)
        results_outer.pack(fill="both", expand=True)

        results_canvas = tk.Canvas(
            results_outer,
            bg=BG_MAIN,
            highlightthickness=0,
            borderwidth=0
        )
        results_scrollbar = ttk.Scrollbar(
            results_outer,
            orient="vertical",
            command=results_canvas.yview
        )
        results_canvas.configure(yscrollcommand=results_scrollbar.set)
        results_scrollbar.pack(side="right", fill="y")
        results_canvas.pack(side="left", fill="both", expand=True)

        self.image_search_results_area = tk.Frame(results_canvas, bg=BG_MAIN)
        results_window = results_canvas.create_window(
            (0, 0),
            window=self.image_search_results_area,
            anchor="nw"
        )

        self.image_search_results_area.bind(
            "<Configure>",
            lambda event: results_canvas.configure(
                scrollregion=results_canvas.bbox("all")
            )
        )
        results_canvas.bind(
            "<Configure>",
            lambda event: results_canvas.itemconfigure(
                results_window,
                width=event.width
            )
        )

        self.image_search_results_canvas = results_canvas

    def show_image_search_results(self, results):
        if not hasattr(self, "image_search_results_area"):
            return

        for widget in self.image_search_results_area.winfo_children():
            widget.destroy()

        self.thumbnail_refs = []

        if not results:
            tk.Label(
                self.image_search_results_area,
                text="No similar images found.",
                bg=BG_MAIN,
                fg=TEXT_SECONDARY,
                font=(FONT_FAMILY, 11),
            ).grid(row=0, column=0, sticky="w", padx=10, pady=12)
            return

        self.show_toast(f"✓ Found {len(results)} similar images", "success")
        self.render_cards(
            results,
            show_scores=True,
            parent=self.image_search_results_area,
            scrollable=False
        )

        if hasattr(self, "image_search_results_canvas"):
            self.image_search_results_canvas.configure(
                scrollregion=self.image_search_results_canvas.bbox("all")
            )

    def show_image_results(self, results):
        self.clear_content()
        self.set_page_title("Similar Image Results")
        self.render_cards(results, show_scores=True)

    # ═══════════════════════════════════════════════════
    # ASYNC TASK RUNNER (unchanged logic)
    # ═══════════════════════════════════════════════════
    def start_task(self, status, function, on_success):
        if self.busy:
            messagebox.showinfo(
                "Please wait",
                "Another operation is still running."
            )
            return

        self.busy = True
        self.set_status(status)

        def worker():
            try:
                result = function()
                self.events.put(("success", result, on_success))
            except Exception as error:
                self.events.put(("error", str(error), None))

        threading.Thread(target=worker, daemon=True).start()

    def process_events(self):
        try:
            while True:
                kind, payload, callback = self.events.get_nowait()
                self.busy = False

                if kind == "success":
                    self.set_status("Operation completed.")
                    callback(payload)
                else:
                    self.set_status("Operation failed.")
                    self.show_toast(f"Error: {payload[:80]}", "error")
                    messagebox.showerror("Operation Error", payload)

        except queue.Empty:
            pass

        self.root.after(100, self.process_events)

    def calculate_file_hash(self, file_path):
        """Calculate a SHA-256 hash to identify identical files."""
        digest = hashlib.sha256()

        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

        return digest.hexdigest()

    # ═══════════════════════════════════════════════════
    # IMPORT IMAGES (unchanged logic, new UI)
    # ═══════════════════════════════════════════════════
    def import_images_from_folder(self):
        """Select a folder, import images safely, and optionally index them."""
        self.set_active_nav("import")
        if self.busy:
            messagebox.showinfo(
                "Please wait",
                "Another operation is still running."
            )
            return

        source_folder = filedialog.askdirectory(
            title="Select a folder containing images"
        )

        if not source_folder:
            return

        source = Path(source_folder).resolve()
        destination = IMAGE_FOLDER.resolve()

        supported_extensions = {
            ".jpg", ".jpeg", ".png", ".bmp",
            ".webp", ".tif", ".tiff"
        }

        # Prevent recursively importing the gallery into itself.
        if source == destination:
            messagebox.showinfo(
                "Gallery folder selected",
                "This is already the gallery folder."
            )
            return

        def import_worker():
            destination.mkdir(parents=True, exist_ok=True)

            copied = 0
            duplicates = 0
            invalid = 0
            errors = 0

            for root, dirs, files in os.walk(source, followlinks=False):
                # Avoid hidden subfolders.
                dirs[:] = [
                    folder for folder in dirs
                    if not folder.startswith(".")
                ]

                for filename in files:
                    if Path(filename).suffix.lower() not in supported_extensions:
                        continue

                    source_path = Path(root) / filename

                    try:
                        # Do not follow file symlinks.
                        if source_path.is_symlink() or not source_path.is_file():
                            continue

                        source_hash = self.calculate_file_hash(source_path)

                        # Keep the original filename where possible.
                        target_path = destination / filename

                        if target_path.exists():
                            if target_path.is_file():
                                try:
                                    if self.calculate_file_hash(target_path) == source_hash:
                                        duplicates += 1
                                        continue
                                except OSError:
                                    pass

                            # A different file has the same name.
                            stem = source_path.stem
                            suffix = source_path.suffix
                            counter = 1

                            while True:
                                candidate = destination / (
                                    f"{stem}__{counter}{suffix}"
                                )

                                if candidate.exists():
                                    try:
                                        if (
                                            candidate.is_file()
                                            and self.calculate_file_hash(candidate)
                                            == source_hash
                                        ):
                                            target_path = candidate
                                            duplicates += 1
                                            break
                                    except OSError:
                                        pass

                                    counter += 1
                                    continue

                                target_path = candidate
                                break

                            if target_path.exists():
                                continue

                        # Verify that the file is a readable image.
                        try:
                            with Image.open(source_path) as image:
                                image.verify()
                        except Exception:
                            invalid += 1
                            continue

                        shutil.copy2(source_path, target_path)
                        copied += 1

                    except (OSError, PermissionError):
                        errors += 1

            return {
                "copied": copied,
                "duplicates": duplicates,
                "invalid": invalid,
                "errors": errors
            }

        def import_finished(summary):
            self.set_status("Folder import finished.")

            report = (
                f"Folder import complete.\n\n"
                f"New images copied: {summary['copied']}\n"
                f"Identical files skipped: {summary['duplicates']}\n"
                f"Unreadable images skipped: {summary['invalid']}\n"
                f"Files that could not be imported: {summary['errors']}"
            )

            if summary["copied"] > 0:
                report += "\n\nWould you like to index the imported images now?"

                should_index = messagebox.askyesno(
                    "Import complete",
                    report
                )

                if should_index:
                    self.index_images()
                else:
                    messagebox.showinfo(
                        "Import complete",
                        "Images were copied. Click Index images when ready."
                    )
                self.show_toast(f"✓ {summary['copied']} images imported", "success")
            else:
                messagebox.showinfo("Import complete", report)

        self.start_task(
            "Scanning and importing images...",
            import_worker,
            import_finished
        )

    # ═══════════════════════════════════════════════════
    # INDEX IMAGES (Oracle + HSV + CLIP)
    # ═══════════════════════════════════════════════════
    def index_images(self):
        if self.busy:
            messagebox.showinfo(
                "Please wait",
                "Another operation is still running."
            )
            return

        if not IMAGE_MANAGER.exists():
            messagebox.showerror(
                "Missing file",
                f"Could not find:\n{IMAGE_MANAGER}"
            )
            return

        def run_indexing():
            # Step 1: Index images in Oracle and extract HSV features.
            oracle_result = subprocess.run(
                [sys.executable, "-m", "src.image_manager"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=600,
                check=True
            )

            # Step 2: Synchronize CLIP embeddings incrementally.
            clip_result = subprocess.run(
                [sys.executable, "-m", "src.sync_clip_embeddings"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=1800,
                check=True
            )

            return oracle_result, clip_result

        def indexing_done(results):
            oracle_result, clip_result = results

            # Step 3: Clear the old CLIP embedding cache.
            from src.text_search import refresh_embedding_cache
            refresh_embedding_cache()

            self.refresh_gallery()
            self.show_gallery()

            self.show_toast(
                "✓ Oracle and CLIP indexing completed successfully",
                "success"
            )

            messagebox.showinfo(
                "Indexing complete",
                "Image indexing and CLIP synchronization finished successfully.\n\n"
                "ORACLE / HSV INDEXING:\n"
                + oracle_result.stdout[-1200:]
                + "\n\nCLIP SYNCHRONIZATION:\n"
                + clip_result.stdout[-1800:]
            )

        self.start_task(
            "Indexing images and synchronizing CLIP embeddings...",
            run_indexing,
            indexing_done
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartImageGallery(root)
    root.mainloop()