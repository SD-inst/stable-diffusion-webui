from modules import extra_networks, script_callbacks, sd_models
from modules.processing import StableDiffusionProcessing
from modules.shared import opts

class ExtraNetworkModel(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('model')
        self.model = opts.sd_model_checkpoint

    def activate(self, p: StableDiffusionProcessing, params_list):
        if len(params_list) == 0:
            return

        opts.sd_model_checkpoint = params_list[0].items[0]
        sd_models.reload_model_weights()

    def deactivate(self, p: StableDiffusionProcessing):
        if self.model is None:
            return
        opts.sd_model_checkpoint = self.model
        sd_models.reload_model_weights()

def before_ui():
    extra_networks.register_extra_network(ExtraNetworkModel())

script_callbacks.on_before_ui(before_ui)
