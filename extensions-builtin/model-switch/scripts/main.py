from modules import extra_networks, script_callbacks, sd_models
from modules.processing import StableDiffusionProcessing
from modules.shared import opts
import re

modelPromptCleaner = re.compile('<model:[^>]*>\s*', re.I)
csPromptCleaner = re.compile('<cs:[^>]*>\s*', re.I)

class ExtraNetworkModel(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('model')
        self.model = opts.sd_model_checkpoint

    def activate(self, p: StableDiffusionProcessing, params_list):
        if len(params_list) == 0:
            return

        if p.is_hr_pass and p.hr_checkpoint_name is not None:
            return
        opts.sd_model_checkpoint = params_list[0].items[0]
        sd_models.reload_model_weights()
        p.all_prompts = [modelPromptCleaner.sub('', pr) for pr in p.all_prompts]
        p.main_prompt = modelPromptCleaner.sub('', p.main_prompt)
        p.extra_generation_params["Model hash"] = params_list[0].items[0]
        if len(params_list[0].items) > 1:
            p.extra_generation_params["Model"] = params_list[0].items[1]

    def deactivate(self, p: StableDiffusionProcessing):
        if self.model is None:
            return
        opts.sd_model_checkpoint = self.model
        sd_models.reload_model_weights()

class ExtraNetworkClipSkip(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('cs')
        self.cs = int(opts.CLIP_stop_at_last_layers)

    def activate(self, p: StableDiffusionProcessing, params_list):
        if len(params_list) == 0:
            return

        opts.CLIP_stop_at_last_layers = int(params_list[0].items[0])
        p.extra_generation_params["Clip skip"] = opts.CLIP_stop_at_last_layers
        p.all_prompts = [csPromptCleaner.sub('', pr) for pr in p.all_prompts]
        p.main_prompt = csPromptCleaner.sub('', p.main_prompt)

    def deactivate(self, p: StableDiffusionProcessing):
        opts.CLIP_stop_at_last_layers = self.cs


def before_ui():
    extra_networks.register_extra_network(ExtraNetworkModel())
    extra_networks.register_extra_network(ExtraNetworkClipSkip())

script_callbacks.on_before_ui(before_ui)
