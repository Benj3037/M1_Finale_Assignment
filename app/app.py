import pandas as pd
import pickle
from xgboost import XGBRegressor
import gradio as gr

# loading X2 created from notebook
X2 = pd.read_json('app\X2.json')

# using best model created from notebook 
best_model = pickle.load(open('app\Best_model.pkl','rb'))

# Pipeline from notebook
new_df_num = pd.DataFrame({
    'cap-color': 0,
    'bruises': 0,
    'odor': 7,
    'gill-spacing': 0,
    'gill-color': 0,
    'stalk-shape': 0,
    'stalk-surface-above-ring': 3,
    'stalk-surface-below-ring': 3,
    'stalk-color-below-ring': 7,
    'veil-color': 2,
    'ring-number': 1,
    'ring-type': 4,
    'spore-print-color': 0,
    'population': 3,
    'habitat': 4

}, index = [0])

new_values_num = pd.DataFrame(new_df_num, columns = new_df_num.columns, index=[0])


# function for app
def mushroom_prediction(cap_color=0, bruises=0, odor=7, gill_spacing=0, gill_color=0, stalk_shape=0, stalk_surface_above_ring=3, stalk_surface_below_ring=3, stalk_color_below_ring=7, veil_color=2, ring_number=1, ring_type=4, spore_print_color=0, population=3, habitat=4): #parameter = default_value

  new_df_num = pd.DataFrame({
      'cap-color': [cap_color],
      'bruises': [bruises],
      'odor': [odor],
      'gill-spacing': [gill_spacing],
      'gill-color': [gill_color],
      'stalk-shape': [stalk_shape],
      'stalk-surface-above-ring': [stalk_surface_above_ring],
      'stalk-surface-below-ring': [stalk_surface_below_ring],
      'stalk-color-below-ring': [stalk_color_below_ring],
      'veil-color': [veil_color],
      'ring-number': [ring_number],
      'ring-type': [ring_type],
      'spore-print-color': [spore_print_color],
      'population': [population],
      'habitat': [habitat]
  })

  new_values_num = pd.DataFrame(new_df_num, columns = new_df_num.columns, index=[0])
  line_to_pred = new_values_num

  prediction = best_model.predict(line_to_pred)
  return prediction[0]

# Create a Gradio interface with custom CSS styles
iface = gr.Interface(
    fn=mushroom_prediction,
    inputs=[
                      gr.Dropdown(X2['cap-color'].unique().tolist(), label="cap-color"),
                      gr.Dropdown(X2['bruises'].unique().tolist(), label="bruises"),
                      gr.Dropdown(X2['odor'].unique().tolist(), label="odor"),
                      gr.Dropdown(X2['gill-spacing'].unique().tolist(), label="gill-spacing"),
                      gr.Dropdown(X2['gill-color'].unique().tolist(), label="gill-color"),
                      gr.Dropdown(X2['stalk-shape'].unique().tolist(), label="stalk-shape"),
                      gr.Dropdown(X2['stalk-surface-above-ring'].unique().tolist(), label="stalk-surface-above-ring"),
                      gr.Dropdown(X2['stalk-surface-below-ring'].unique().tolist(), label="stalk-surface-below-ring"),
                      gr.Dropdown(X2['stalk-color-below-ring'].unique().tolist(), label="stalk-color-below-ring"),
                      gr.Dropdown(X2['veil-color'].unique().tolist(), label="veil-color"),
                      gr.Dropdown(X2['ring-number'].unique().tolist(), label="ring-number"),
                      gr.Dropdown(X2['ring-type'].unique().tolist(), label="ring-type"),
                      gr.Dropdown(X2['spore-print-color'].unique().tolist(), label="spore-print-color"),
                      gr.Dropdown(X2['population'].unique().tolist(), label="population"),
                      gr.Dropdown(X2['habitat'].unique().tolist(), label="habitat"),
                     ],
    outputs="text",
    live=True,
    title="Mushroom Predictor",
    description="Predict if mushroom is edible or not based on input features."
  )

# Launch the Gradio app
iface.launch(share=True)