import gradio as gr
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from annoy import AnnoyIndex
import json
import torch
import cv2

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Загружаем Annoy-индекс
embedding_dim = 512
index = AnnoyIndex(embedding_dim, 'angular')
index.load("image_embeddings_256.ann")

# Загружаем метаданные
with open("annoy_metadata.json", 'r') as f:
    metadata = json.load(f)


def find_box(input_image):
    image = cv2.imread(input_image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (7, 7), 0)

    # Применим метод Canny для нахождения контуров
    edges = cv2.Canny(gray, 70, 130)

    # Увеличим контуры для лучшей детекции
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Найдем контуры на изображении
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Нарисуем контуры
    output = image.copy()

    h_glob, w_glob, _ = image.shape
    s_glob = h_glob * w_glob
    max_s = 0
    found = None

    for contour in contours:
        # Получим прямоугольник, обрамляющий контур
        x, y, w, h = cv2.boundingRect(contour)
        s = h * w
        # Фильтруем по размеру, чтобы убрать мелкие шумы
        if w > 50 and h > 50 and s <= 0.7 * s_glob and 0.3 < w / h < 3 and s > max_s:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            max_s = s
            found = image[y:y + h, x:x + w]

    if found is None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(found, cv2.COLOR_BGR2RGB)


def run(text, input_image, n_pictures, detect_flag):
    if not input_image:
        return

    if detect_flag:
        image = Image.fromarray(find_box(input_image)).convert("RGB")
    else:
        image = Image.open(input_image).convert("RGB")

    # Преобразуем изображение в вектор с помощью CLIP
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)
        image_embedding = image_embedding.cpu().numpy().flatten()

    final_embedding = image_embedding

    text = text.strip()
    if text:
        inputs = processor(text=text, return_tensors="pt")
        with torch.no_grad():
            text_embedding = model.get_text_features(**inputs)
            text_embedding = text_embedding.cpu().numpy().flatten()

        final_embedding += text_embedding * 0.4

    indices = index.get_nns_by_vector(final_embedding, n_pictures)

    files = []
    for idx in indices:
        # Индексы хранятся как строки, поэтому преобразуем их в строку
        metadata_entry = metadata.get(str(idx))
        files.append(f"{metadata_entry['directory']}/{metadata_entry['filename']}")

    return load_dataset("DoomAI/doomai", data_files=files)["train"]["image"]


# Gradio
with gr.Blocks(
        theme=gr.themes.Default(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.orange)) as demo:
    gr.Markdown("# DoomAI")
    with gr.Row():
        with gr.Column():
            # Ввод изображения
            image_input = gr.Image(label="Картинка", type="filepath", sources=['upload', 'clipboard'])

            # Дополнительные параметры
            pictures_slider = gr.Slider(
                label="Количество картинок", minimum=1.0, maximum=20.0, value=10, step=1,
            )
            checkbox = gr.Checkbox(
                False, label="Удаление фона скриншота"
            )
            prompt_input = gr.Textbox(
                label="Уточнение изображения на английском (опционально)", placeholder="A photo of a cute dog"
            )

            # Кнопка
            generate_button = gr.Button("Найти", variant='primary')

        with gr.Column():
            # Вывод
            gallery = gr.Gallery(label='Галерея', columns=4, rows=5, show_share_button=False)

    # При нажатии
    generate_button.click(
        run,
        inputs=[
            prompt_input,
            image_input,
            pictures_slider,
            checkbox
        ],
        outputs=gallery
    )

# Запуск приложения
demo.launch()
