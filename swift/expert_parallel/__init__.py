# Copyright (c) ModelScope Contributors. All rights reserved.

from .ep import EP_SHARDED_FLAG, ExpertParallel, expert_parallel
from .fsdp_patch import patch_fsdp_ignore_ep_params
