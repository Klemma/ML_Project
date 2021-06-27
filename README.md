# [Данные, обученные модели, презентация](https://drive.google.com/drive/folders/14GvCTtqMJvdqfS19SvmbJ0Cxny0Qe_wZ?usp=sharing)

# Де-лемматизация текста
**Участники проекта:**<br>
* Клемин И.Р. ИДБ-18-09<br>
* Смирнов А.С. ИДБ-18-09
## Краткое описание проекта
В лемматизированное или частично лемматизированное предложение, подставим подлежащее, укажем его атрибуты (род и число), а также время сказуемого. Генеративная модель детектирует подсказку в виде этого подлежащего, его атрибутов и времени сказуемого и делемматизирует предложение (из начальных форм слов строит грамматически значимое предложение).

## Идея реализации проекта
Имеется датасет, состоящий из предложений, записанных полностью или частично в начальных формах, и грамматически значимых предложений. Из грамматически значимых предложений выделяется контекст: подлежащее, его род и число, а также время сказуемого. Используемая модель: finetuned **seq2seq** на основе **mbart-large-50**. Вход модели строится следующим образом: **"подлежащее [gender] [tense] [number] [sos] лемматизированное предложение [eos]" (1)**. При обучении, в качестве входной последовательности используется шаблон **(1)**, а в качестве метки соответствующее грамматически значимое предложение.

## Инструкция по установке
1. Клонировать проект на локальный компьютер.
2. Создать в корне проекта папку **models**, если ее там нет.
3. Скачать [архив](https://drive.google.com/file/d/1OZKX-2AWH5Rg60uIYhq82w3rKb4AtJ6o/view?usp=sharing) и разархивировать его в папку **models**.
4. Установить необходимые зависимости, указанные в [requirements.txt](requirements.txt).
5. Из-под корня проекта выполнить в консоли команду: _python -m src.application_.

**Предупреждение:**<br>
Приложение требует **~5-6 ГБ** оперативной памяти.
