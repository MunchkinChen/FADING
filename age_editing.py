import os
import pandas as pd
import argparse
from diffusers import StableDiffusionPipeline, DDIMScheduler

from FADING_util import util
from p2p import *
from null_inversion import *

#%%
parser = argparse.ArgumentParser()

parser.add_argument('--FFHQ_path', required=True, help='Path to FFHQ dataset')
parser.add_argument('--FFHQ_label_path', required=True, help='Path to ffhq_aging_labels.csv')
parser.add_argument('--FFHQ_id', type=int, required=True, help='Number of image')
parser.add_argument('--specialized_path', default="runwayml/stable-diffusion-v1-5", help='Path to specialized diffusion model')
parser.add_argument('--save_aged_dir', default='./outputs', help='Path to save outputs')
parser.add_argument('--target_ages', nargs='+', default=[10, 20, 40, 60, 80], type=int, help='Target age values')

args = parser.parse_args()

#%%
FFHQ_path = args.FFHQ_path
FFHQ_id = args.FFHQ_id
save_aged_dir = args.save_aged_dir
FFHQ_label_path = args.FFHQ_label_path
specialized_path = args.specialized_path
target_ages = args.target_ages


image_path = os.path.join(FFHQ_path,f'{(FFHQ_id // 1000)*1000:05d}',f'{FFHQ_id:05d}.png')
FFHQ_label_csv = pd.read_csv(FFHQ_label_path)

if not os.path.exists(save_aged_dir):
    os.makedirs(save_aged_dir)

#%% get initial age and gender
age_group_mapping = {
    '0-2': 1,
    '3-6': 5,
    '7-9': 8,
    '10-14': 12,
    '15-19': 17,
    '20-29': 25,
    '30-39': 35,
    '40-49': 45,
    '50-69': 60,
    '70-120': 80
}
age_group = FFHQ_label_csv.loc[FFHQ_id, 'age_group']
gender = FFHQ_label_csv.loc[FFHQ_id, 'gender']

age_init = age_group_mapping[age_group]
gt_gender = int(gender == 'female')

person_placeholder = util.get_person_placeholder(age_init, gt_gender)
inversion_prompt = f"photo of {age_init} year old {person_placeholder}"
print(f"inversion prompt: {inversion_prompt}")

#%% load specialized diffusion model
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                          clip_sample=False, set_alpha_to_one=False,
                          steps_offset=1)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
g_cuda = torch.Generator(device=device)

ldm_stable = StableDiffusionPipeline.from_pretrained(specialized_path,
    scheduler=scheduler,
    safety_checker=None).to(device)
tokenizer = ldm_stable.tokenizer

#%% null text inversion
null_inversion = NullInversion(ldm_stable)
(image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, inversion_prompt,
                                                                      offsets=(0,0,0,0), verbose=True)
#%% age editing
for age_new in target_ages:
    new_person_placeholder = util.get_person_placeholder(age_new, gt_gender)
    new_prompt = inversion_prompt.replace(person_placeholder, new_person_placeholder)

    new_prompt = new_prompt.replace(str(age_init),str(age_new))
    blend_word = (((str(age_init),person_placeholder,), (str(age_new),new_person_placeholder,)))
    is_replace_controller = True

    prompts = [inversion_prompt, new_prompt]
    print("p2p prompts:", prompts)

    cross_replace_steps = {'default_': .8,}
    self_replace_steps = .5

    eq_params = {"words": (str(age_new)), "values": (1,)} # amplify attention to the word by *1
    # eq_params = None

    controller = make_controller(prompts, is_replace_controller, cross_replace_steps, self_replace_steps,
                                 tokenizer, blend_word, eq_params)

    images, _ = p2p_text2image(ldm_stable, prompts, controller, generator=g_cuda.manual_seed(0),
                               latent=x_t, uncond_embeddings=uncond_embeddings)

    # res = ptp_utils.view_images(images)
    # util.mydisplay(res)

    reconstructed_path = os.path.join(save_aged_dir,f'{FFHQ_id:05d}_{age_init}_reconstructed.png')
    reconstructed_img = images[0]

    if not os.path.exists(reconstructed_path):
        reconstructed_img_pil = Image.fromarray(reconstructed_img)
        reconstructed_img_pil.save(reconstructed_path)

    new_img = images[-1]
    new_img_pil = Image.fromarray(new_img)
    new_img_pil.save(os.path.join(save_aged_dir,f'{FFHQ_id:05d}_{age_new}.png'))
