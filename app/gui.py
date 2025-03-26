import gradio as gr
from inference.predict import HandwritingRecognizer

def create_gui(model_path):
    recognizer = HandwritingRecognizer(model_path)
    
    def recognize(image):
        if image is None:
            return ""
        return recognizer.predict_char(image)
    
    interface = gr.Interface(
        fn=recognize,
        inputs=gr.Sketchpad(shape=(280, 280),  # 10x tamaño original
        outputs="label",
        live=True,
        title="Reconocimiento de Escritura Manual",
        description="Dibuje una letra o número en el área blanca"
    )
    
    return interface

if __name__ == "__main__":
    from config import MODEL_DIR
    gui = create_gui(f"{MODEL_DIR}/cnn_model.h5")
    gui.launch()