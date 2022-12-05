import streamlit as st
import streamlit_drawable_canvas as st_canvas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

###  this app showcases the tflite model with streamlit

#  load the model
@st.cache(allow_output_mutation=True)
class model:
    def __init__(self) -> None:
        self.interpreter = tf.lite.Interpreter(model_path="cnn.tflite")

mymodel = model()
interpreter = mymodel.interpreter
interpreter.allocate_tensors()

#  get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#  canvas parameters
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 13)
stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FFFFFF")
bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")
bg_image = None
drawing_mode = st.sidebar.selectbox("Drawing tool:", ("freedraw", "line", "rect", "circle", "transform"))
realtime_update = st.sidebar.checkbox("Update in realtime", True)

#  create a canvas component
canvas_result = st_canvas.st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=bg_image,
    update_streamlit=realtime_update,
    height=500,
    width=500,
    drawing_mode=drawing_mode,
    key="canvas",
)


#  always run this

#  get the image from the canvas
image = canvas_result.image_data.astype(np.float32)
#  reshape the image
image = cv2.resize(image, (28, 28))
#  normalize the image
image = image / 255.0
#  display the image as grayscale
st.image(image, channels="BGR")
#  remove the alpha channel
image = image[:, :, :3]
#  grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#  add a batch dimension
image = image[tf.newaxis, ...]
image = image[tf.newaxis, ...]

#  set the input tensor
interpreter.set_tensor(input_details[0]["index"], image)
#  run the inference
interpreter.invoke()
#  get the output tensor
output_data = interpreter.get_tensor(output_details[0]["index"])
#  get the prediction
prediction = np.argmax(output_data)
#  display the prediction
st.write("# The number is: ", prediction)
#  display the probability of all the classes as a bar chart
st.bar_chart(output_data[0]+np.min(output_data[0]))
