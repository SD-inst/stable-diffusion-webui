import os

from modules import shared, ui_extra_networks, sd_models
from modules.ui_extra_networks_checkpoints_user_metadata import CheckpointUserMetadataEditor


class ExtraNetworksPageCheckpoints(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Checkpoints')

    def refresh(self):
        shared.refresh_checkpoints()

    def create_item(self, name, index=None, enable_filter=True):
        checkpoint: sd_models.CheckpointInfo = sd_models.checkpoint_aliases.get(name)
        path, ext = os.path.splitext(checkpoint.filename)
        return {
            "name": checkpoint.name_for_extra,
            "filename": checkpoint.filename,
            "shorthash": checkpoint.shorthash,
            "preview": self.find_preview(path),
            "description": self.find_description(path),
            "search_term": self.search_terms_from_path(checkpoint.filename) + " " + (checkpoint.sha256 or ""),
            "onclick": '"' + f"""var ta = get_uiCurrentTabContent().querySelector('#override_settings_control textarea');
                ta.value = '{checkpoint.shorthash}|{checkpoint.name_for_extra}';
                ta.dispatchEvent(new Event('input'));
                get_uiCurrentTabContent().querySelector('.extra-networks > .tab-nav > button:nth-child(1)').click()""" + '"',
            "local_preview": f"{path}.{shared.opts.samples_format}",
            "metadata": checkpoint.metadata,
            "sort_keys": {'default': index, **self.get_sort_keys(checkpoint.filename)},
        }

    def list_items(self):
        names = list(sd_models.checkpoints_list)
        for index, name in enumerate(names):
            yield self.create_item(name, index)

    def allowed_directories_for_previews(self):
        return [v for v in [shared.cmd_opts.ckpt_dir, sd_models.model_path] if v is not None]

    def create_user_metadata_editor(self, ui, tabname):
        return CheckpointUserMetadataEditor(ui, tabname, self)
