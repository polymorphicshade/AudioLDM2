from huggingface_hub import hf_hub_download
import torch
import os
import random
import numpy as np

import gradio as gr
from audioldm2 import text_to_audio, build_model

os.environ["TOKENIZERS_PARALLELISM"] = "true"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

default_checkpoint = "audioldm_48k"
audioldm = None
current_model_name = None

def text2audio(
    text,
    duration,
    guidance_scale,
    random_seed,
    n_candidates,
    model_name=default_checkpoint,
):
    global audioldm, current_model_name
    torch.set_float32_matmul_precision("high")

    random_seed = int(random_seed)
    n_candidates = int(n_candidates)

    if random_seed == 0:
        random_seed = random.randint(1, 100000000)
        print(f"Random seed (0 supplied) generated: {random_seed}")

    if audioldm is None or model_name != current_model_name:
        print(f"Loading model: {model_name}")
        audioldm = build_model(model_name=model_name)
        audioldm.to(DEVICE)
        current_model_name = model_name
    
    if "48k" in model_name:
        latent_t_per_second = 12.8
        sample_rate = 48000
    else:
        latent_t_per_second = 25.6
        sample_rate = 16000

    waveform_raw = text_to_audio(
        latent_diffusion=audioldm,
        text=text,
        seed=random_seed,
        duration=duration,
        guidance_scale=guidance_scale,
        n_candidate_gen_per_text=n_candidates,
        latent_t_per_second=latent_t_per_second,
    )  # [bs, 1, samples]

    # the output of text_to_audio is a list of [1, samples] numpy arrays.
    if len(waveform_raw) == 1:
        return (sample_rate, waveform_raw[0][0])
    else:
        return [(sample_rate, wave[0]) for wave in waveform_raw]

iface = gr.Blocks()

with iface:
    with gr.Column():
        textbox = gr.Textbox(
            value="A forest of wind chimes singing a soothing melody in the breeze.",
            label="Input your text here. Your text is important for the audio quality. Please ensure it is descriptive by using more adjectives."
        )

        with gr.Accordion("Click to modify detailed configurations", open=False):
            seed = gr.Slider( 
                minimum=0,
                maximum=100000000,
                value=0,
                step=1,
                label="Change this value (any integer number) will lead to a different generation result. Enter 0 for a random result.",
            )
            duration = gr.Slider(
                5, 15, value=10, step=2.5, label="Duration (seconds)"
            )
            guidance_scale = gr.Slider(
                0,
                6,
                value=3.5,
                step=0.5,
                label="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
            )
            n_candidates = gr.Slider(
                1,
                3,
                value=3,
                step=1,
                label="Automatic quality control. This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with heavier computation",
            )
            model_name = gr.Dropdown(
                ["audioldm_48k", "audioldm_crossattn_flant5", "audioldm2-full"], value="audioldm_48k", label="Model"
            )
        
        outputs = gr.Audio(label="Output", elem_id="output-audio", type="numpy")

        btn = gr.Button("Submit")
        btn.css_class = "gr-btn-full-width"

    btn.click(
        text2audio,
        inputs=[textbox, duration, guidance_scale, seed, n_candidates, model_name],
        outputs=[outputs],
        api_name="text2audio",
    )
    gr.HTML(
        """
            <div class="acknowledgements">
            <p>Essential Tricks for Enhancing the Quality of Your Generated Audio</p>
            <p>1. Try to use more adjectives to describe your sound. For example: "A man is speaking clearly and slowly in a large room" is better than "A man is speaking". This can make sure AudioLDM 2 understands what you want.</p>
            <p>2. Try to use different random seeds, which can affect the generation quality significantly sometimes.</p>
            <p>3. It's better to use general terms like 'man' or 'woman' instead of specific names for individuals or abstract objects that humans may not be familiar with, such as 'mummy'.</p>
            </div>
            """
    )

iface.queue().launch(
    server_name="0.0.0.0",
    server_port=7860,
    debug=False,
    share=False
)