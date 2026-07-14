#!/usr/bin/env python3
"""Shared window-building helpers for the LLM Medical Cases programs.

All programs are Microsoft Windows-friendly Tkinter applications: Tkinter
ships with Python on Windows, so nothing extra needs to be installed.
This module is imported lazily (inside each program's main()) so the
non-GUI logic and the test suites keep working on machines without Tk.
"""

import queue
import threading
import traceback

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, scrolledtext

PAD = 8


def make_root(title, width=980, height=680):
    root = tk.Tk()
    root.title("LLM Medical Cases - " + title)
    root.geometry("{}x{}".format(width, height))
    root.minsize(720, 480)
    try:
        ttk.Style().theme_use("vista")  # native look on Windows
    except tk.TclError:
        pass
    return root


def bring_to_front(window):
    """Force a window above everything else - during a browser run the
    automation browser covers the screen, and a question dialog opening
    BEHIND it looks exactly like the program freezing (the worker waits
    forever for an answer the user cannot see)."""
    try:
        window.deiconify()
        window.lift()
        window.attributes("-topmost", True)
        window.focus_force()
        window.bell()
    except Exception:
        pass


def drop_topmost(window):
    try:
        window.attributes("-topmost", False)
    except Exception:
        pass


def show_error(title, message, parent=None):
    messagebox.showerror(title, message, parent=parent)


def ask_int(parent, title, question, minvalue=1):
    return simpledialog.askinteger(title, question, parent=parent, minvalue=minvalue)


def ask_string(parent, title, question):
    value = simpledialog.askstring(title, question, parent=parent)
    return value.strip() if value else ""


def pick_from_list(parent, title, question, options):
    """Modal list picker; returns the chosen option or None."""
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()
    tk.Label(dialog, text=question, justify="left").pack(padx=PAD, pady=(PAD, 4))
    box = tk.Listbox(dialog, height=min(12, max(4, len(options))), width=60)
    for option in options:
        box.insert("end", option)
    box.selection_set(0)
    box.pack(padx=PAD, pady=4, fill="both", expand=True)
    chosen = []

    def accept(_event=None):
        selection = box.curselection()
        if selection:
            chosen.append(options[selection[0]])
        dialog.destroy()

    box.bind("<Double-Button-1>", accept)
    row = tk.Frame(dialog)
    row.pack(pady=(4, PAD))
    tk.Button(row, text="OK", width=10, command=accept).pack(side="left", padx=4)
    tk.Button(row, text="Cancel", width=10, command=dialog.destroy).pack(side="left", padx=4)
    bring_to_front(dialog)
    dialog.wait_window()
    return chosen[0] if chosen else None


def ask_choice(parent, title, message, options):
    """Modal dialog with one button per option.

    options: list of (key, label). Returns the chosen key, or None if the
    window is closed.
    """
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()
    tk.Label(dialog, text=message, justify="left", wraplength=520).pack(
        padx=PAD * 2, pady=(PAD * 2, PAD)
    )
    chosen = []

    def pick(key):
        chosen.append(key)
        dialog.destroy()

    buttons = tk.Frame(dialog)
    buttons.pack(pady=(0, PAD * 2))
    for key, label in options:
        tk.Button(
            buttons, text=label, width=max(12, len(label) + 2),
            command=lambda k=key: pick(k),
        ).pack(side="left", padx=4)
    bring_to_front(dialog)
    dialog.wait_window()
    return chosen[0] if chosen else None


class LogBox:
    """A read-only scrolling log pane."""

    def __init__(self, parent, height=12):
        self.widget = scrolledtext.ScrolledText(
            parent, height=height, state="disabled", wrap="word"
        )

    def pack(self, **kwargs):
        self.widget.pack(**kwargs)
        return self

    def log(self, message):
        self.widget.configure(state="normal")
        self.widget.insert("end", message + "\n")
        self.widget.see("end")
        self.widget.configure(state="disabled")

    def clear(self):
        self.widget.configure(state="normal")
        self.widget.delete("1.0", "end")
        self.widget.configure(state="disabled")


def make_table(parent, columns, widths=None):
    """A ttk.Treeview table with a vertical scrollbar.

    columns: list of (key, heading). Returns the tree; its container frame
    is tree.master (pack/grid that).
    """
    frame = tk.Frame(parent)
    tree = ttk.Treeview(
        frame, columns=[key for key, _ in columns], show="headings", selectmode="extended"
    )
    for i, (key, heading) in enumerate(columns):
        tree.heading(key, text=heading)
        width = (widths or {}).get(key, 120)
        tree.column(key, width=width, anchor="w", stretch=True)
    scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    tree.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")
    return tree


def readonly_text(parent, height=10):
    widget = scrolledtext.ScrolledText(parent, height=height, wrap="word", state="disabled")
    return widget


def set_text(widget, content):
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", content)
    widget.configure(state="disabled")


class ThreadUI:
    """Given to worker threads so long jobs can log and ask questions
    without touching Tk from the wrong thread. Requests are queued and the
    main loop answers them; ask_* calls block the worker until the user
    responds."""

    def __init__(self, task):
        self._task = task
        self.stop_requested = False

    def log(self, message):
        self._task.queue.put(("log", message, None))

    def _ask(self, kind, payload):
        done = threading.Event()
        request = {"kind": kind, "payload": payload, "done": done, "answer": None}
        self._task.queue.put(("ask", request, None))
        done.wait()
        return request["answer"]

    def tell(self, title, message):
        """Blocking information dialog (worker waits until OK is clicked)."""
        self._ask("info", (title, message))

    def ask_yes_no(self, title, message):
        return bool(self._ask("yesno", (title, message)))

    def ask_choice(self, title, message, options):
        return self._ask("choice", (title, message, options))


class BackgroundTask:
    """Runs one function on a worker thread; log lines and dialogs are
    marshalled onto the Tk main loop via a queue pumped with after()."""

    def __init__(self, root, log_callback):
        self.root = root
        self.log_callback = log_callback
        self.queue = queue.Queue()
        self.thread = None
        self.on_done = None

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, fn, on_done=None):
        """fn(ui) runs on the worker; on_done(result, error) runs on the
        main loop afterwards."""
        if self.running:
            raise RuntimeError("a task is already running")
        self.on_done = on_done
        ui = ThreadUI(self)

        def work():
            try:
                result = fn(ui)
                self.queue.put(("done", result, None))
            except BaseException as error:  # surfaced in the window, never lost
                self.queue.put(("done", None, error))

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()
        self.root.after(100, self._pump)
        return ui

    def _pump(self):
        finished = False
        while True:
            try:
                kind, payload, extra = self.queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self.log_callback(payload)
            elif kind == "ask":
                self._answer(payload)
            elif kind == "done":
                finished = True
                if self.on_done is not None:
                    if extra is not None and not isinstance(extra, Exception):
                        # KeyboardInterrupt etc.
                        extra = RuntimeError(str(extra))
                    self.on_done(payload, extra)
        if not finished and self.running:
            self.root.after(100, self._pump)
        elif not finished:
            # Thread ended without a done message (shouldn't happen); keep
            # pumping briefly in case the queue is trailing.
            self.root.after(100, self._pump) if not self.queue.empty() else None

    def _answer(self, request):
        kind = request["kind"]
        payload = request["payload"]
        # A worker question must never hide behind the automation browser.
        bring_to_front(self.root)
        try:
            if kind == "info":
                messagebox.showinfo(payload[0], payload[1], parent=self.root)
                request["answer"] = True
            elif kind == "yesno":
                request["answer"] = messagebox.askyesno(
                    payload[0], payload[1], parent=self.root
                )
            elif kind == "choice":
                request["answer"] = ask_choice(self.root, payload[0], payload[1], payload[2])
        except Exception:
            traceback.print_exc()
            request["answer"] = None
        finally:
            drop_topmost(self.root)
            request["done"].set()
