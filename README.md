# DoomAI (Поиск смысловых копий изображений)

Наше решение представляет собой готовый продукт для поиска семантических аналогов изображений — веб-приложение на базе технологий машинного обучения, позволяющее находить содержательно схожие изображения.

Приложение переводит изображения в векторное пространство с помощью open-source библиотеки CLIP, что позволяет находить аналогичные изображения по принципу векторной близости. Мы предлагаем подборку из 10–20 похожих изображений, основываясь на расчете косинусного расстояния между вектором целевого изображения и векторами изображений в базе данных.

**Технические особенности:** ЯП python, храним датасет на hugging face, храним векторы через библиотеку spotify annoy, gradio для веб-интерфейса

Уникальность решения заключается в высокой точности и масштабируемости, а также возможности текстом уточнить сюжет картинки. Примененные технологии позволяют эффективно работать с разнообразными и крупными наборами данных, что создает широкие перспективы для решения будущих задач.
