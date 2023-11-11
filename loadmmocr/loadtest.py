import copy
import importlib
import os.path as osp
import re
import warnings
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, Union)

import numpy as np
import torch
import torch.nn as nn
from rich.progress import track

from mmengine.config import Config, ConfigDict
from mmengine.config.utils import MODULE2PACKAGE
from mmengine.dataset import pseudo_collate
from mmengine.device import get_device
from mmengine.fileio import (get_file_backend, isdir, join_path,
                             list_dir_or_file, load)
from mmengine.logging import print_log
from mmengine.registry import FUNCTIONS, MODELS, VISUALIZERS, DefaultScope
from mmengine.runner.checkpoint import (_load_checkpoint,
                                        _load_checkpoint_to_model)
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer


#my path
from UDUP_pp.Allconfig import Path_Config


InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray, torch.Tensor]
InputsType = Union[InputType, Sequence[InputType]]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict]]
ConfigType = Union[Config, ConfigDict]
ModelType = Union[dict, ConfigType, str]

class InferencerMeta(ABCMeta):
    """Check the legality of the inferencer.

    All Inferencers should not define duplicated keys for
    ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs`` and
    ``postprocess_kwargs``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.preprocess_kwargs, set)
        assert isinstance(self.forward_kwargs, set)
        assert isinstance(self.visualize_kwargs, set)
        assert isinstance(self.postprocess_kwargs, set)

        all_kwargs = (
            self.preprocess_kwargs | self.forward_kwargs
            | self.visualize_kwargs | self.postprocess_kwargs)

        assert len(all_kwargs) == (
            len(self.preprocess_kwargs) + len(self.forward_kwargs) +
            len(self.visualize_kwargs) + len(self.postprocess_kwargs)), (
                f'Class define error! {self.__name__} should not '
                'define duplicated keys for `preprocess_kwargs`, '
                '`forward_kwargs`, `visualize_kwargs` and '
                '`postprocess_kwargs` are not allowed.')

#Config DBnet=path: /workspace/mmocr/configs/textdet/dbnet/dbnet_resnet50-oclip_1200e_icdar2015.py
class BaseInferencer(metaclass=InferencerMeta):

    preprocess_kwargs: set = set()
    forward_kwargs: set = set()
    visualize_kwargs: set = set()
    postprocess_kwargs: set = set()

    def __init__(self,
                 model: Union[ModelType, str, None] = None,
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: Optional[str] = None,
                 show_progress: bool = True) -> None:
        # if scope is None:
        #     default_scope = DefaultScope.get_current_instance()
        #     if default_scope is not None:
        #         scope = default_scope.scope_name
        self.scope = 'mmocr'
        # Load config to cfg
        cfg: ConfigType
        if isinstance(model, str):
            if osp.isfile(model):
                cfg = Config.fromfile(model)
            else:
                # Load config and weights from metafile. If `weights` is
                # assigned, the weights defined in metafile will be ignored.
                cfg, _weights = self._load_model_from_metafile(model)
                if weights is None:
                    weights = _weights
        elif isinstance(model, (Config, ConfigDict)):
            cfg = copy.deepcopy(model)
        elif isinstance(model, dict):
            cfg = copy.deepcopy(ConfigDict(model))
        elif model is None:
            if weights is None:
                raise ValueError(
                    'If model is None, the weights must be specified since '
                    'the config needs to be loaded from the weights')
            cfg = ConfigDict()
        else:
            raise TypeError('model must be a filepath or any ConfigType'
                            f'object, but got {type(model)}')

        if device is None:
            device = get_device()

        self.model = self._init_model(cfg, weights, device)  # type: ignore
        # self.pipeline = self._init_pipeline(cfg)
        # self.collate_fn = self._init_collate(cfg)
        # self.visualizer = self._init_visualizer(cfg)
        # self.cfg = cfg
        # self.show_progress = show_progress

    def __call__(
        self,
        inputs: InputsType,
        return_datasamples: bool = False,
        batch_size: int = 1,
        **kwargs,
    ) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            batch_size (int): Batch size. Defaults to 1.
            **kwargs: Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results.
        """
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        ori_inputs = self._inputs_to_list(inputs)
        inputs = self.preprocess(
            ori_inputs, batch_size=batch_size, **preprocess_kwargs)
        preds = []
        for data in (track(inputs, description='Inference')
                     if self.show_progress else inputs):
            preds.extend(self.forward(data, **forward_kwargs))
        visualization = self.visualize(
            ori_inputs, preds,
            **visualize_kwargs)  # type: ignore  # noqa: E501
        results = self.postprocess(preds, visualization, return_datasamples,
                                   **postprocess_kwargs)
        return results
    def _load_model_from_metafile(self, model: str) -> Tuple[Config, str]:
        """Load config and weights from metafile.

        Args:
            model (str): model name defined in metafile.

        Returns:
            Tuple[Config, str]: Loaded Config and weights path defined in
            metafile.
        """
        model = model.lower()

        repo_or_mim_dir = Path_Config.mmocr_path
        for model_cfg in BaseInferencer._get_models_from_metafile(repo_or_mim_dir):
            model_name = model_cfg['Name'].lower()
            model_aliases = model_cfg.get('Alias', [])
            if isinstance(model_aliases, str):
                model_aliases = [model_aliases.lower()]
            else:
                model_aliases = [alias.lower() for alias in model_aliases]
            if (model_name == model or model in model_aliases):
                cfg = Config.fromfile(
                    osp.join(repo_or_mim_dir, model_cfg['Config']))
                weights = model_cfg['Weights']
                weights = weights[0] if isinstance(weights, list) else weights
                return cfg, weights
        raise ValueError(f'Cannot find model: {model} in {self.scope}')





    def _init_model(
        self,
        cfg: ConfigType,
        weights: Optional[str],
        device: str = 'cpu',
    ) -> nn.Module:
        """Initialize the model with the given config and checkpoint on the
        specific device.

        Args:
            cfg (ConfigType): Config containing the model information.
            weights (str, optional): Path to the checkpoint.
            device (str, optional): Device to run inference. Defaults to 'cpu'.

        Returns:
            nn.Module: Model loaded with checkpoint.
        """
        checkpoint: Optional[dict] = None
        checkpoint = _load_checkpoint(weights, map_location='cpu')



        # Delete the `pretrained` field to prevent model from loading the
        # the pretrained weights unnecessarily.
        if cfg.model.get('pretrained') is not None:
            del cfg.model.pretrained

        model = MODELS.build(cfg.model)
        model.cfg = cfg
        self._load_weights_to_model(model, checkpoint, cfg)
        model.to(device)
        model.eval()
        return model

    def _load_weights_to_model(self, model: nn.Module,
                               checkpoint: Optional[dict],
                               cfg: Optional[ConfigType]) -> None:
        """Loading model weights and meta information from cfg and checkpoint.

        Subclasses could override this method to load extra meta information
        from ``checkpoint`` and ``cfg`` to model.

        Args:
            model (nn.Module): Model to load weights and meta information.
            checkpoint (dict, optional): The loaded checkpoint.
            cfg (Config or ConfigDict, optional): The loaded config.
        """
        if checkpoint is not None:
            _load_checkpoint_to_model(model, checkpoint)
        else:
            warnings.warn('Checkpoint is not loaded, and the inference '
                          'result is calculated by the randomly initialized '
                          'model!')
    @staticmethod
    def _get_models_from_metafile(dir: str):
        """Load model config defined in metafile from package path.

        Args:
            dir (str): Path to the directory of Config. It requires the
                directory ``Config``, file ``model-index.yml`` exists in the
                ``dir``.

        Yields:
            dict: Model config defined in metafile.
        """
        meta_indexes = load(osp.join(dir, 'model-index.yml'))
        for meta_path in meta_indexes['Import']:
            # meta_path example: mmcls/.mim/configs/conformer/metafile.yml
            meta_path = osp.join(dir, meta_path)
            metainfo = load(meta_path)
            yield from metainfo['Models']



BaseI=BaseInferencer(model='DBNet',scope='mmocr',device='cuda')
print("as")
