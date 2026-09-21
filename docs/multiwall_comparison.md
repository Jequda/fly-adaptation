# Шаг 9. Повторяемость маршрута через две стены

Теперь весь multiwall-опыт повторяется с независимыми начальными популяциями
и комнатами. Это проверка алгоритма поиска, а не еще один replay удачного
мозга.

## 1. Быстрый запуск для знакомства

```powershell
uv run python -m experiments.compare_multiwall --generations 5 --population 16 --episodes 16 --validation-episodes 16 --holdout-episodes 16 --max-steps 160 --workers 4 --replicates 2 --open-viewer
```

Используй его, чтобы увидеть переключение seed и структуру файлов. Не делай по
нему вывод о воспроизводимости.

## 2. Полный протокол

```powershell
uv run python -m experiments.compare_multiwall --generations 50 --population 80 --episodes 32 --validation-episodes 32 --holdout-episodes 64 --max-steps 160 --workers 4 --replicates 5 --seed 7 --seed-step 10 --open-viewer
```

Каждый повтор получает отдельные веса, мутации, training, validation и
holdout. Наборы seed между повторами не пересекаются.

## Критерии PASS

Все условия должны выполниться одновременно:

```text
mean holdout              >= 80%
запуски с holdout >= 70%  >= 4 из 5
mean orientation gap      <= 15 п.п.
mean same/zigzag gap      <= 20 п.п.
mean crossed both         >= 85%
mean stuck                <= 15%
```

Например, нулевой разрыв категорий при нулевом успехе не дает PASS, потому что
общий holdout также входит в логическое `AND`.

## Наш результат

```text
mean holdout = 99.69%
range = 98.44-100%
запуски >= 70% = 5/5
mean crossed both = 99.69%
вердикт = PASS
```

По основному навыку результат устойчив. При этом анализ неудач обнаружил
полезную деталь: seed `17` решил все сцены, но имел `25%` stuck и в среднем
`11.39` столкновения. Успех и качество траектории не являются одной метрикой.

## Как исследовать разброс

В `comparison.csv` отсортируй seed по holdout, collisions и stuck. Затем
открой соответствующий `runs/seed-*/viewer.html` и сравни `Success` и
`Failure`. Даже когда eat rate равен `100%`, viewer может показать хрупкую
стратегию, которая тратит много шагов на столкновения.

`comparison-viewer.html` показывает лучших победителей в общей showcase-
комнате. Он удобен для визуального сравнения, но численный рейтинг рассчитан
по всем holdout-сценам.

## Файлы

- `report.md` содержит итоговые критерии;
- `comparison.csv` содержит одну строку на seed;
- `categories.csv` содержит 16 типов маршрутов;
- `runs/seed-*/` хранит подробности каждого повтора;
- `comparison-viewer.html` показывает независимых победителей вместе.

После устойчивой навигации мы возвращаемся к памяти в более строгой форме:
агенту надо запомнить не положение, а [правило выбора цели](target_choice_experiment.md).
