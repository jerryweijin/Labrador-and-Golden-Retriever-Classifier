from fastai.vision.all import *
import gradio as gr

learn = load_learner('model.pkl')

categories=('Golden Retriever', 'Labrador')
def predict(img):
    pred,idx,prob = learn.predict(img)
    return dict(zip(categories, map(float, prob)))

# Define a gradio interface
image = gr.inputs.Image((224,224))
label = gr.outputs.Label()

interface = gr.Interface(fn=predict, inputs=image, outputs=label)
interface.launch()
