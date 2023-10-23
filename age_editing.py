import os
import argparse
from diffusers import StableDiffusionPipeline, DDIMScheduler

from FADING_util import util
from p2p import *
from null_inversion import *

#%%
parser = argparse.ArgumentParser()

parser.add_argument('--image_path', required=True, help='Path to input image')
parser.add_argument('--age_init', required=True, type=int, help='Specify the initial age')
parser.add_argument('--gender', required=True, choices=["female", "male"], help="Specify the gender ('female' or 'male')")
parser.add_argument('--specialized_path', required=True, help='Path to specialized diffusion model')
parser.add_argument('--save_aged_dir', default='./outputs', help='Path to save outputs')
parser.add_argument('--target_ages', nargs='+', default=[10, 20, 40, 60, 80], type=int, help='Target age values')

args = parser.parse_args()

#%%
image_path = args.image_path
age_init = args.age_init
gender = args.gender
save_aged_dir = args.save_aged_dir
specialized_path = args.specialized_path
target_ages = args.target_ages

if not os.path.exists(save_aged_dir):
    os.makedirs(save_aged_dir)

gt_gender = int(gender == 'female')
person_placeholder = util.get_person_placeholder(age_init, gt_gender)
inversion_prompt = f"photo of {age_init} year old {person_placeholder}"

input_img_name = image_path.split('/')[-1].split('.')[-2]

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
    print(f'Age editing with target age {age_new}...')
    new_person_placeholder = util.get_person_placeholder(age_new, gt_gender)
    new_prompt = inversion_prompt.replace(person_placeholder, new_person_placeholder)

    new_prompt = new_prompt.replace(str(age_init),str(age_new))
    blend_word = (((str(age_init),person_placeholder,), (str(age_new),new_person_placeholder,)))
    is_replace_controller = True

    prompts = [inversion_prompt, new_prompt]

    cross_replace_steps = {'default_': .8,}
    self_replace_steps = .5

    eq_params = {"words": (str(age_new)), "values": (1,)}

    controller = make_controller(prompts, is_replace_controller, cross_replace_steps, self_replace_steps,
                                 tokenizer, blend_word, eq_params)

    images, _ = p2p_text2image(ldm_stable, prompts, controller, generator=g_cuda.manual_seed(0),
                               latent=x_t, uncond_embeddings=uncond_embeddings)

    new_img = images[-1]
    new_img_pil = Image.fromarray(new_img)
    new_img_pil.save(os.path.join(save_aged_dir,f'{input_img_name}_{age_new}.png'))
