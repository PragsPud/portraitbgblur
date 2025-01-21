import cv2
import mediapipe as mp
import numpy as np
import gradio as gr

mp_selfie = mp.solutions.selfie_segmentation

def pr_segment(image,conf=0.5,ksize1=40,ksize2=60):
  with mp_selfie.SelfieSegmentation(model_selection=0) as selfie:
    res = selfie.process(image)
    mask = np.stack((res.segmentation_mask,)*3, axis=-1) > conf

    mask = np.where(mask, np.full(image.shape,(255,255,255),dtype = np.uint8),np.zeros(image.shape, dtype=np.uint8))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        person_cont = max(contours, key = cv2.contourArea)
        mask = cv2.drawContours(np.zeros(image.shape, dtype=np.uint8), [person_cont,], -1, (255,255,255), thickness=cv2.FILLED)
    else:
      mask = np.zeros(image.shape, dtype=np.uint8)

    return np.where(mask == (255,255,255), image, cv2.blur(image, (ksize1,ksize2)))
      
with gr.Blocks() as webapp:
  with gr.Row():
    with gr.Column():
      img = gr.Image(label = "Input", sources=['upload','clipboard','webcam'])
      conf = gr.Slider(0, 0.99, value=0.5, label="Confidence")
      k_size1 = gr.Number(value=40, label = "Kernel Size X")
      k_size2 = gr.Number(value=60, label = "Kernel Size Y")
    with gr.Column():
      out = gr.Image(label = "Output", streaming=True)
  btn = gr.Button("Blur Background")
  btn.click(pr_segment, inputs=[img,conf,k_size1, k_size2], outputs=[out,])
webapp.launch(share=False,debug=False)