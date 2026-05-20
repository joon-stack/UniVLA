# UniVLA LeRobot BYOP Plugin On Robot Server

이 패키지는 Hugging Face LeRobot의 Bring Your Own Policies 방식에 맞춘 `lerobot_policy_univla` plugin입니다.

현재 상태:

- LeRobot BYOP package로 설치 가능
- `PreTrainedConfig.register_subclass("univla")` 등록
- `UniVLAPolicy`, `UniVLAConfig`, `make_univla_pre_post_processors` 제공
- dummy `predict_action_chunk()`로 `[B, 10, 7]` action chunk smoke 가능
- real `dummy=false`에서 HF UniVLA VLA, processor, action decoder, dataset stats 로드
- LeRobot batch의 `observation.images.primary`, `observation.state`, `task`를 UniVLA 입력으로 변환
- action decoder가 normalized action chunk `[B, 10, 7]`를 반환하고 LeRobot postprocessor가 action unnormalize

제한:

- 현재 real inference는 primary image + language + optional proprio 경로입니다.
- wrist image를 VLA generation에 같이 넣는 multi-view generation 경로는 아직 별도 구현 대상입니다.
- proprio는 LeRobot preprocessor에서 normalize하지 않고, policy 내부에서 UniVLA cache 학습과 동일하게 `dataset_statistics.json`의 `proprio mean/std`로 normalize합니다.

즉, robot server에는 이 BYOP plugin을 설치하고, exported artifact 폴더 하나를 `pretrained_name_or_path`로 넘기는 구조입니다.

## 1. 로봇 서버 Python 확인

`policy_server`를 실행할 바로 그 Python을 잡습니다.

```bash
export LEROBOT_PY=/path/to/robot/venv/bin/python
```

LeRobot 설치 위치:

```bash
export LEROBOT_ROOT=$($LEROBOT_PY -c 'import lerobot, pathlib; print(pathlib.Path(lerobot.__file__).resolve().parent)')
echo "$LEROBOT_ROOT"
```

## 2. BYOP Plugin 설치

로봇 서버에 `UniVLA-hyper` repo가 있다고 가정합니다.

```bash
cd /path/to/UniVLA-hyper

$LEROBOT_PY -m pip install --no-build-isolation -e vla-scripts/lerobot_policy_univla
```

설치 확인:

```bash
$LEROBOT_PY - <<'PY'
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.configs import PreTrainedConfig
register_third_party_plugins()
print("univla" in PreTrainedConfig.get_known_choices())
print(PreTrainedConfig.get_choice_class("univla"))
PY
```

정상이라면 `True`와 `lerobot_policy_univla.configuration_univla.UniVLAConfig`가 나옵니다.

## 3. Async Policy Server Compatibility Patch

현재 LeRobot `policy_server.py`는 `register_third_party_plugins()`를 호출하지 않고, `SUPPORTED_POLICIES` 정적 리스트로 먼저 막습니다.

그래서 robot `policy_server`까지 쓰려면 아래 compatibility patch가 필요합니다.

```bash
cd /path/to/UniVLA-hyper

$LEROBOT_PY vla-scripts/lerobot_policy_univla/patch_lerobot_async_server.py \
  --lerobot-root "$LEROBOT_ROOT"
```

이 패치가 하는 일:

- `lerobot/async_inference/constants.py`의 `SUPPORTED_POLICIES`에 `"univla"` 추가
- `lerobot/async_inference/policy_server.py`가 policy instruction 처리 전에 `register_third_party_plugins()` 호출

이건 BYOP plugin 자체를 LeRobot core에 넣는 패치가 아닙니다. async server가 third-party policy를 볼 수 있게 해주는 작은 호환 패치입니다.

## 4. Smoke Test

```bash
cd /path/to/UniVLA-hyper

$LEROBOT_PY vla-scripts/lerobot_policy_univla/smoke_univla_byop.py
```

정상 출력:

```text
UniVLA LeRobot BYOP plugin smoke passed.
```

이 smoke는 다음을 확인합니다.

- `register_third_party_plugins()` 후 `univla` registry 등록
- `get_policy_class("univla")` 동작
- `make_univla_pre_post_processors()` discovery 동작
- dummy action chunk shape `[1, 10, 7]`
- config/preprocessor/postprocessor 저장 후 `from_pretrained()` 재로드

## 5. Real UniVLA Policy Artifact 만들기

학습 산출물이 아래처럼 있다고 가정합니다.

```text
/path/to/hf_univla_export/
├── config.json
├── dataset_statistics.json
├── tokenizer.model
├── ...
└── action_decoder-10000.pt
```

artifact 생성:

```bash
cd /path/to/UniVLA-hyper

$LEROBOT_PY vla-scripts/lerobot_policy_univla/export_univla_policy_artifact.py \
  --output-dir /path/to/outputs/univla_lerobot_policy_real \
  --vla-path /path/to/hf_univla_export \
  --action-decoder-path /path/to/hf_univla_export/action_decoder-10000.pt \
  --dataset-statistics-path /path/to/hf_univla_export/dataset_statistics.json \
  --dataset-name joon_stack \
  --univla-repo-root /path/to/UniVLA-hyper \
  --image-key observation.images.primary \
  --state-key observation.state \
  --task-key task \
  --window-size 10 \
  --n-action-steps 10 \
  --latent-action-token-len 5 \
  --device cuda
```

`dataset_statistics.json` 안에 dataset key가 여러 개 있으면 `--dataset-name`은 반드시 그중 하나와 정확히 맞아야 합니다.
`--latent-action-token-len`은 checkpoint와 맞아야 합니다. hfrad/euclidean-5token은 `5`, UniVLA stage2는 `4`입니다. 생략하면 `vla_path/config.json` 또는 `eval_meta.json`에서 자동 추론합니다.

생성된 `/path/to/outputs/univla_lerobot_policy_real` 폴더에는 다음이 저장됩니다.

- `config.json`: `type=univla`, `dummy=false`, VLA/action decoder/stats 경로
- `policy_preprocessor.json`
- `policy_preprocessor.safetensors`
- `policy_postprocessor.json`
- `policy_postprocessor.safetensors`

이 폴더를 `policy_server` / `robot_client`의 `pretrained_name_or_path`로 넘기면 됩니다.

```bash
--policy.type univla \
--pretrained-name-or-path /path/to/outputs/univla_lerobot_policy_real
```

## 6. Dummy Policy Artifact 만들기

통신/로드/shape만 확인하려면 dummy artifact를 만듭니다.

```bash
cd /path/to/UniVLA-hyper

$LEROBOT_PY - <<'PY'
from pathlib import Path
import torch

import lerobot_policy_univla  # noqa: F401
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

out = Path("outputs/univla_lerobot_dummy_policy")
out.mkdir(parents=True, exist_ok=True)

image_key = "observation.images.primary"
state_dim = 7
action_dim = 7
cfg_cls = PreTrainedConfig.get_choice_class("univla")
cfg = cfg_cls(
    input_features={
        image_key: PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
        OBS_STATE: PolicyFeature(FeatureType.STATE, (state_dim,)),
    },
    output_features={ACTION: PolicyFeature(FeatureType.ACTION, (action_dim,))},
    device="cuda",
    dummy=True,
    window_size=10,
    n_action_steps=10,
    image_key=image_key,
)

stats = {
    OBS_STATE: {"mean": torch.zeros(state_dim), "std": torch.ones(state_dim)},
    ACTION: {"q01": torch.full((action_dim,), -1.0), "q99": torch.full((action_dim,), 1.0)},
}

pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
cfg.save_pretrained(out)
pre.save_pretrained(out)
post.save_pretrained(out)
print(out.resolve())
PY
```

이 경로를 `robot_client`의 `--pretrained-name-or-path` 또는 대응 config에 넣어 policy_server load path를 확인할 수 있습니다.

주의: dummy policy는 action이 전부 0입니다. 실제 로봇 동작용이 아닙니다.

## 7. 에러별 확인

### `Policy type univla not supported`

async compatibility patch가 안 먹은 겁니다.

```bash
$LEROBOT_PY -c 'from lerobot.async_inference.constants import SUPPORTED_POLICIES; print(SUPPORTED_POLICIES)'
```

여기에 `univla`가 있어야 합니다.

### `Policy type 'univla' is not available`

plugin package가 설치되지 않았거나 `policy_server`가 plugin registration을 안 한 겁니다.

```bash
$LEROBOT_PY - <<'PY'
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.configs import PreTrainedConfig
register_third_party_plugins()
print(PreTrainedConfig.get_known_choices())
PY
```

`univla`가 보여야 합니다.

### `model.safetensors not found`

`UniVLAPolicy.from_pretrained()`가 아니라 기본 `PreTrainedPolicy.from_pretrained()`가 호출되는 상황일 가능성이 큽니다.

확인:

```bash
$LEROBOT_PY - <<'PY'
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.policies.factory import get_policy_class
register_third_party_plugins()
cls = get_policy_class("univla")
print(cls)
print(cls.from_pretrained)
PY
```

`lerobot_policy_univla.modeling_univla.UniVLAPolicy`가 나와야 합니다.
