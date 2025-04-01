# Linear Programming Generator 🚀

Генератор задач линейного программирования с автоматическим созданием условий и решений в форматах LaTeX/PDF. 
Идеальный инструмент для преподавателей и студентов!

[![GitHub Releases](https://img.shields.io/github/v/release/yourusername/linear-programming-generator)](https://github.com/yourusername/linear-programming-generator/releases)

## ✨ Возможности 
- Генерация задач с **целочисленными решениями**
- Настройка параметров через GUI:
  - Количество вершин многоугольника
  - Ограничения на координаты и коэффициенты
  - Количество шагов симплекс-метода
- Экспорт в LaTeX с визуализацией
- Автоматическая компиляция в PDF
- Поддержка Windows, Linux и macOS

## 📥 Релизы
Готовые сборки для всех ОС доступны в разделе [Releases](https://github.com/MrKustic/linear-programming-generator/releases):
- `Windows.exe` — Portable-версия
- `Linux` — ELF-бинарник
- `MacOS` — Universal-приложение

*Для работы сборок не требуется установка Python!*

## ⚙️ Установка (для разработки)
```bash
git clone https://github.com/MrKustic/linear-programming-generator
cd linear-programming-generator
pip install -r requirements.txt
```

## 🎯 Компиляция в PDF
1. Убедитесь, что pdflatex доступен в PATH
2. В GUI отметьте опцию "Необходимо создать pdf файлы на основе файлов tex"
3. Или скомпилируйте вручную:
```bash
pdflatex generated_problem.tex
```
