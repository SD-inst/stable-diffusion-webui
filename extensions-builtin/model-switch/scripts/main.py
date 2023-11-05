from modules import extra_networks, script_callbacks
from modules.processing import StableDiffusionProcessing
from modules.shared import opts
import re

csPromptCleaner = re.compile('<cs:[^>]*>\s*', re.I)

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
    extra_networks.register_extra_network(ExtraNetworkClipSkip())

script_callbacks.on_before_ui(before_ui)
