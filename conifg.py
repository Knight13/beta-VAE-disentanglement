from keras.models import load_model, Model
from src.common import sample_layer, utils


model_path = "saved_models/deepmind_enc_vae27112018.hdf5"
vae_model = load_model(model_path, custom_objects={'SampleLayer': sample_layer.SampleLayer})
encoder_output_layer = 'sampling_layer'
decoder_input_layer = 'decoder_inp'

enc_model = Model(inputs=[vae_model.input], outputs=[vae_model.get_layer('sampling_layer').output])

model_splitter = utils.SplitModel(parent_model=vae_model)

start_idx = model_splitter.get_layer_idx_by_name(layername=decoder_input_layer)
end_idx = len(vae_model.layers)

dec_model = model_splitter.split_model(start=start_idx, end=end_idx)
