import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import platform
import time

from main import GenerateLinearProgrammingProblems

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Генератор задач ЛП")
        self.geometry("1000x900")  # Увеличили высоту для новых ошибок

        self.create_widgets()
        self.setup_styles()

        # Требования для валидации
        self.requirements = [
            (self.n, self.n_error, 3, 10**9, True),
            (self.cnt_steps_first_entry, self.cnt_steps_first_error, 0, 10**9, False),
            (self.cnt_steps_second_entry, self.cnt_steps_second_error, 1, 10**9, True),
            (self.ineq_max_value_entry, self.ineq_max_error, 1, 10**18, True),
            (self.func_max_value_entry, self.func_max_error, 1, 10**18, True),
            (self.cnt_statements_entry, self.cnt_statements_error, 1, 10**9, True)
        ]

        self.additional_requirements = [
            (self.save_path, self.save_path_error, self.validate_path),
            (self.statements_filename, self.statements_filename_error, self.validate_filename),
            (self.solutions_filename, self.solutions_filename_error, self.validate_filename)
        ]

        self.setup_validations()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.configure("TButton", padding=6)
        self.style.configure("TEntry", padding=5)
        self.style.configure("TLabel", padding=5)
        self.style.configure("Invalid.TEntry", foreground="red")
        self.style.configure("Valid.TEntry", foreground="black")

    def get_default_path(self):
        home = os.path.expanduser("~")
        if platform.system() == "Windows":
            return os.path.join(home, "Documents")
        else:
            return os.path.join(home, "Desktop")

    def create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Числовые поля
        ttk.Label(main_frame, text="Количество вершин в многоугольнике: ").grid(row=0, column=0, sticky="w")
        self.n = ttk.Entry(main_frame, style="Valid.TEntry")
        self.n.grid(row=0, column=1, sticky="ew", pady=5, padx=5)
        self.n_error = ttk.Label(main_frame, text="", foreground="red")
        self.n_error.grid(row=1, column=0, columnspan=3, sticky="w", padx=5)

        # Чекбокс и связанные поля
        self.is_zero_allowed = tk.IntVar(value=0)
        ttk.Checkbutton(
            main_frame,
            text="(0, 0) является допустимым решением",
            variable=self.is_zero_allowed,
            command=self.toggle_cnt_steps_first_field
        ).grid(row=2, column=0, columnspan=3, pady=5, padx=5, sticky="w")

        self.cnt_steps_first_label = ttk.Label(main_frame,
                                               text="Количество шагов для достижения допустимого решения\n(оставьте пустым, если это не имеет значения):")
        self.cnt_steps_first_label.grid(row=3, column=0, sticky='w', pady=5, padx=5)
        self.cnt_steps_first_entry = ttk.Entry(main_frame, style="Valid.TEntry")
        self.cnt_steps_first_entry.grid(row=3, column=1, sticky='ew', pady=5, padx=5)
        self.cnt_steps_first_error = ttk.Label(main_frame, text="", foreground="red")
        self.cnt_steps_first_error.grid(row=4, column=0, columnspan=3, sticky="w", padx=5)

        ttk.Label(main_frame, text="Количество шагов для достижения оптимального решения:").grid(row=5, column=0, sticky='w', pady=5, padx=5)
        self.cnt_steps_second_entry = ttk.Entry(main_frame, style="Valid.TEntry")
        self.cnt_steps_second_entry.grid(row=5, column=1, sticky='ew', pady=5, padx=5)
        self.cnt_steps_second_error = ttk.Label(main_frame, text="", foreground="red")
        self.cnt_steps_second_error.grid(row=6, column=0, columnspan=3, sticky="w", padx=5)

        # Ограничения
        ttk.Label(main_frame, text="Верхнее ограничение на значения координат многогранников:").grid(row=7, column=0, sticky='w', pady=5, padx=5)
        self.ineq_max_value_entry = ttk.Entry(main_frame, style="Valid.TEntry")
        self.ineq_max_value_entry.grid(row=7, column=1, sticky='ew', pady=5, padx=5)
        self.ineq_max_error = ttk.Label(main_frame, text="", foreground="red")
        self.ineq_max_error.grid(row=8, column=0, columnspan=3, sticky="w", padx=5)

        ttk.Label(main_frame, text="Верхнее ограничение на абсолютные значения коэффициентов в функционале:").grid(row=9, column=0, sticky='w', pady=5, padx=5)
        self.func_max_value_entry = ttk.Entry(main_frame, style="Valid.TEntry")
        self.func_max_value_entry.grid(row=9, column=1, sticky='ew', pady=5, padx=5)
        self.func_max_error = ttk.Label(main_frame, text="", foreground="red")
        self.func_max_error.grid(row=10, column=0, columnspan=3, sticky="w", padx=5)

        ttk.Label(main_frame, text="Количество вариантов работ:").grid(row=11, column=0, sticky='w', pady=5, padx=5)
        self.cnt_statements_entry = ttk.Entry(main_frame, style="Valid.TEntry")
        self.cnt_statements_entry.grid(row=11, column=1, sticky='ew', pady=5, padx=5)
        self.cnt_statements_error = ttk.Label(main_frame, text="", foreground="red")
        self.cnt_statements_error.grid(row=12, column=0, columnspan=3, sticky="w", padx=5)

        # Дополнительные настройки
        self.create_pdf = tk.IntVar(value=0)
        ttk.Checkbutton(
            main_frame,
            text="Необходимо создать pdf файлы на основе файлов tex",
            variable=self.create_pdf
        ).grid(row=13, column=0, columnspan=3, pady=5, padx=5, sticky='w')

        # Пути и названия файлов
        ttk.Label(main_frame, text="Папка для сохранения файлов").grid(row=14, column=0, sticky='w', pady=5, padx=5)
        self.save_path = tk.StringVar(value=self.get_default_path())
        tk.Entry(main_frame, textvariable=self.save_path).grid(row=14, column=1, sticky='ew', pady=5, padx=5)
        self.save_path_error = ttk.Label(main_frame, text="", foreground="red")
        tk.Button(main_frame, text="Browse...", command=self.select_path).grid(row=14, column=2, sticky='w', pady=5, padx=5)
        self.save_path_error.grid(row=15, column=0, columnspan=3, sticky='w', padx=5)

        ttk.Label(main_frame, text="Название файла с условиями (без расширения):").grid(row=16, column=0, sticky='w', pady=5, padx=5)
        self.statements_filename = tk.StringVar(value="statements")
        tk.Entry(main_frame, textvariable=self.statements_filename).grid(row=16, column=1, sticky='ew', padx=5, pady=5)
        self.statements_filename_error = ttk.Label(main_frame, text="", foreground="red")
        self.statements_filename_error.grid(row=17, column=0, columnspan=3, sticky='w', padx=5)

        ttk.Label(main_frame, text="Название файла с решениями (без расширения):").grid(row=18, column=0, sticky='w', pady=5, padx=5)
        self.solutions_filename = tk.StringVar(value="solutions")
        tk.Entry(main_frame, textvariable=self.solutions_filename).grid(row=18, column=1, sticky='ew', padx=5, pady=5)
        self.solutions_filename_error = ttk.Label(main_frame, text="", foreground="red")
        self.solutions_filename_error.grid(row=19, column=0, columnspan=3, sticky='w', padx=5)

        # Управление
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=20, column=0, columnspan=4, pady=15, sticky="ew")
        self.run_btn = ttk.Button(control_frame, text="Начать генерацию", command=self.start_process)
        self.run_btn.pack(pady=5)
        self.progress = ttk.Progressbar(control_frame, mode="indeterminate", length=900)

        main_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(0, weight=1)

    def setup_validations(self):
        # Валидация числовых полей
        for entry, error_label, min_val, max_val, is_req in self.requirements:
            entry.bind("<FocusOut>",
                       lambda e, el=entry, err=error_label, mn=min_val, mx=max_val, rq=is_req:
                       self.validate_number(el, err, mn, mx, rq))
            entry.bind("<KeyRelease>",
                       lambda e, el=entry, err=error_label, mn=min_val, mx=max_val, rq=is_req:
                       self.validate_number(el, err, mn, mx, rq))

        # Валидация дополнительных полей
        for var, error_label, validator in self.additional_requirements:
            var.trace_add(
                "write",
                lambda *args, v=var, el=error_label, val=validator: val(v, el)
            )
            # Первоначальная проверка
            validator(var, error_label)

    def validate_number(self, entry_widget, error_label, min_val, max_val, req):
        value = entry_widget.get()
        valid = True
        error_text = ""

        if value:
            try:
                num = int(value)
                if not (min_val <= num <= max_val):
                    error_text = f"Допустимый диапазон: {min_val}-{max_val}"
                    valid = False
            except ValueError:
                error_text = "Введите целое число"
                valid = False
        elif req:
            valid = False
            error_text = "Поле обязательно для заполнения"

        error_label.config(text=error_text)
        entry_widget.config(style="Valid.TEntry" if valid else "Invalid.TEntry")
        return valid

    def validate_path(self, var, error_label):
        path = var.get()
        valid = True
        error_text = ""

        if not path:
            error_text = "Путь обязателен для заполнения"
            valid = False
        else:
            if not os.path.exists(path):
                try:
                    os.makedirs(path, exist_ok=True)
                except Exception as e:
                    error_text = f"Ошибка создания директории: {str(e)}"
                    valid = False
            if valid and not os.access(path, os.W_OK):
                error_text = "Нет прав на запись в директорию"
                valid = False

        error_label.config(text=error_text)
        return valid

    def validate_filename(self, var, error_label):
        name = var.get().strip()
        valid = bool(name)
        error_text = "Название файла не может быть пустым" if not valid else ""

        error_label.config(text=error_text)
        return valid

    def toggle_cnt_steps_first_field(self):
        if self.is_zero_allowed.get() == 0:
            self.cnt_steps_first_label.grid()
            self.cnt_steps_first_entry.grid()
            self.cnt_steps_first_entry.config(state="normal")
        else:
            self.cnt_steps_first_label.grid_remove()
            self.cnt_steps_first_entry.grid_remove()
            self.cnt_steps_first_entry.delete(0, "end")
            self.cnt_steps_first_entry.config(state="disabled")

    def select_path(self):
        path = filedialog.askdirectory(initialdir=self.save_path.get())
        if path:
            self.save_path.set(path)
            self.validate_path(self.save_path, self.save_path_error)

    def start_process(self):
        # Проверка всех валидаций
        validations = [
            *[self.validate_number(*args) for args in self.requirements],
            *[validator(var, error) for (var, error, validator) in self.additional_requirements]
        ]

        if not all(validations):
            messagebox.showerror("Ошибка", "Исправьте неверно заполненные поля")
            return

        try:
            params = {
                "n": int(self.n.get()),
                "m": 2,
                "is_zero_allowed": int(self.is_zero_allowed.get()),
                "cnt_steps_first": 0 if self.is_zero_allowed.get() or not self.cnt_steps_first_entry.get() else int(self.cnt_steps_first_entry.get()),
                "cnt_steps_second": int(self.cnt_steps_second_entry.get()),
                "ineq_max_value": int(self.ineq_max_value_entry.get()),
                "func_max_value": int(self.func_max_value_entry.get()),
                "cnt_statements": int(self.cnt_statements_entry.get()),
                "path_to_statements": os.path.join(self.save_path.get(), self.statements_filename.get()),
                "path_to_solutions": os.path.join(self.save_path.get(), self.solutions_filename.get()),
                "only_latex": not self.create_pdf.get()
            }

            self.operation_timeout = 60
            self.operation_completed = False

            self.toggle_ui(False)
            self.progress.pack(pady=5)
            self.progress.start()

            self.thread = threading.Thread(
                target=self.execute_main_function,
                kwargs=params,
                daemon=True
            )
            self.thread.start()
            self.monitor_thread()

        except ValueError as e:
            messagebox.showerror("Ошибка преобразования данных", str(e))

    def execute_main_function(self, **kwargs):
        try:
            self.operation_start_time = time.time()
            GenerateLinearProgrammingProblems(**kwargs)
            if not self.operation_completed:
                self.after(0, lambda: self.show_success())
        except Exception as e:
            if not self.operation_completed:
                self.after(0, lambda: self.show_error(str(e)))
        finally:
            self.operation_completed = True
            self.after(0, self.toggle_ui, True)

    def monitor_thread(self):
        if self.thread.is_alive():
            elapsed = time.time() - self.operation_start_time
            if elapsed > self.operation_timeout:
                self.operation_completed = True
                self.progress.stop()
                self.progress.pack_forget()
                self.after(0, lambda: messagebox.showerror(
                    "Ошибка",
                    f"Превышено время выполнения. Попробуйте увеличить ограничения на коэффициенты, уменьшить количество вершин в многоугольнике или изменить кол-во шагов для достижения допустимого решения."
                ))
                self.toggle_ui(True)
            else:
                self.after(100, self.monitor_thread)
        else:
            self.progress.stop()
            self.progress.pack_forget()
            self.toggle_ui(True)

    def show_success(self):
        messagebox.showinfo(
            "Успешное выполнение",
            f"Генерация завершена успешно!\n"
        )

    def show_error(self, error_msg):
        error_type = "Ошибка записи" if "PermissionError" in error_msg else "Ошибка генерации"
        messagebox.showerror(
            error_type,
            f"Произошла ошибка во время выполнения:\n{error_msg}"
        )

    def toggle_ui(self, state: bool):
        self.run_btn.config(state="normal" if state else "disabled")

if __name__ == "__main__":
    app = Application()
    app.mainloop()