# Deprecated: Use `vla-scripts/lerobot_policy_univla/README_ROBOT_SERVER.md`

이 문서는 예전 내장형 site-packages 패치 방식입니다. Hugging Face LeRobot Bring Your Own Policies 방식에 맞춘 현재 구현은 아래 경로를 쓰세요.

```text
vla-scripts/lerobot_policy_univla/
vla-scripts/lerobot_policy_univla/README_ROBOT_SERVER.md
```

예전 내장형 패치를 이미 적용했다면 cleanup만 이 폴더에서 사용하면 됩니다.

```bash
$LEROBOT_PY vla-scripts/lerobot_univla_patch/remove_builtin_univla_patch.py \
  --lerobot-root "$LEROBOT_ROOT"
```

---

# Legacy UniVLA LeRobot Patch On Robot Server

이 문서는 로봇이 연결된 서버의 기존 LeRobot 환경에 `policy_type=univla`를 추가하는 절차입니다.

현재 패치는 **학습 코드가 아니라 runtime adapter smoke용 패치**입니다. 즉, LeRobot의 `policy_server` / `robot_client`가 `univla` policy type을 받아들이고, `predict_action_chunk()` 인터페이스로 `[B, 10, 7]` action chunk를 주고받을 수 있는지 확인하는 단계입니다.

## 1. 전제

로봇 서버에서 아래 두 가지가 같아야 합니다.

- `policy_server`를 실행할 때 쓰는 Python/venv
- 이 패치를 적용할 LeRobot 설치 경로

예를 들어 `policy_server`를 `/path/to/venv/bin/python -m lerobot...`로 띄운다면, 패치도 반드시 `/path/to/venv/bin/python` 기준으로 적용해야 합니다.

## 2. LeRobot 경로 확인

로봇 서버에서 `policy_server`에 쓰는 venv를 활성화하거나, 해당 Python을 직접 지정합니다.

```bash
export LEROBOT_PY=/path/to/robot/venv/bin/python

$LEROBOT_PY -c 'import lerobot, pathlib; print(pathlib.Path(lerobot.__file__).resolve().parent)'
```

출력 예:

```text
/path/to/robot/venv/lib/python3.12/site-packages/lerobot
```

이 출력값이 `LEROBOT_ROOT`입니다.

```bash
export LEROBOT_ROOT=$($LEROBOT_PY -c 'import lerobot, pathlib; print(pathlib.Path(lerobot.__file__).resolve().parent)')
echo "$LEROBOT_ROOT"
```

## 3. 패치 적용

`UniVLA-hyper` repo가 로봇 서버에 있어야 합니다.

```bash
cd /path/to/UniVLA-hyper

$LEROBOT_PY vla-scripts/lerobot_univla_patch/apply_lerobot_univla_patch.py \
  --lerobot-root "$LEROBOT_ROOT"
```

정상 출력:

```text
Applied UniVLA LeRobot patch to /path/to/robot/venv/lib/python3.12/site-packages/lerobot
```

이 스크립트가 하는 일:

- `lerobot/policies/univla/` 추가
- `lerobot/policies/factory.py`에 `univla` config/model/processor 등록
- `lerobot/async_inference/constants.py`의 `SUPPORTED_POLICIES`에 `univla` 추가

스크립트는 같은 환경에 여러 번 실행해도 중복 등록되지 않도록 작성되어 있습니다.

## 4. Smoke Test

패치 직후 같은 Python으로 smoke test를 실행합니다.

```bash
cd /path/to/UniVLA-hyper

$LEROBOT_PY vla-scripts/lerobot_univla_patch/smoke_univla_policy.py
```

정상 출력:

```text
UniVLA LeRobot policy smoke passed.
```

이 smoke test가 확인하는 것:

- `get_policy_class("univla")` 동작
- `make_policy_config("univla")` 동작
- preprocessor/postprocessor 생성
- dummy `predict_action_chunk()` shape: `[1, 10, 7]`
- config + processor 저장 후 `from_pretrained()` 재로드
- `SUPPORTED_POLICIES`에 `univla` 포함

## 5. Dummy Policy Artifact 만들기

아직 진짜 UniVLA checkpoint가 없을 때 policy_server plumbing만 확인하려면, dummy artifact를 하나 만들면 됩니다.

```bash
cd /path/to/UniVLA-hyper

$LEROBOT_PY - <<'PY'
from pathlib import Path
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import make_policy_config, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

out = Path("outputs/univla_lerobot_dummy_policy")
out.mkdir(parents=True, exist_ok=True)

image_key = "observation.images.primary"
state_dim = 7
action_dim = 7

cfg = make_policy_config(
    "univla",
    input_features={
        image_key: PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
        OBS_STATE: PolicyFeature(FeatureType.STATE, (state_dim,)),
    },
    output_features={
        ACTION: PolicyFeature(FeatureType.ACTION, (action_dim,)),
    },
    device="cuda",
    dummy=True,
    window_size=10,
    n_action_steps=10,
    image_key=image_key,
)

stats = {
    OBS_STATE: {
        "q01": torch.full((state_dim,), -1.0),
        "q99": torch.full((state_dim,), 1.0),
    },
    ACTION: {
        "q01": torch.full((action_dim,), -1.0),
        "q99": torch.full((action_dim,), 1.0),
    },
}

pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
cfg.save_pretrained(out)
pre.save_pretrained(out)
post.save_pretrained(out)
print(out.resolve())
PY
```

이 경로를 `policy_server`의 `pretrained_name_or_path`로 넘기면 dummy action chunk까지는 나옵니다.

주의: dummy policy는 실제 로봇을 움직이기 위한 policy가 아닙니다. action이 전부 0이므로, 통신/로드/shape 확인용입니다.

## 6. 실제 UniVLA Checkpoint 연결 시 해야 할 일

현재 `UniVLAPolicy(dummy=False)`의 real checkpoint loading은 아직 구현되어 있지 않습니다. 다음 단계에서 아래를 채워야 합니다.

- UniVLA VLA checkpoint load
- action decoder checkpoint load
- image tensor `[B, C, H, W]` -> UniVLA image processor 입력 변환
- `task` string/list 처리
- latent action token 생성
- action decoder로 `[B, 10, 7]` chunk 생성
- dataset statistics 기반 action unnormalization과 LeRobot postprocessor 계약 확인

즉, 이 패치는 LeRobot 쪽 문을 여는 작업이고, 실제 UniVLA inference body는 다음 구현 대상입니다.

## 7. 자주 나는 문제

### `Policy type 'univla' is not available`

원인:

- 패치를 다른 venv에 적용함
- `policy_server`는 A venv로 실행 중인데 패치는 B venv에 적용함

확인:

```bash
which python
python -c 'import lerobot, pathlib; print(pathlib.Path(lerobot.__file__).resolve().parent)'
python -c 'from lerobot.policies.factory import get_policy_class; print(get_policy_class("univla"))'
```

### `univla is not in SUPPORTED_POLICIES`

원인:

- `lerobot/async_inference/constants.py`에 패치가 안 먹음

확인:

```bash
python -c 'from lerobot.async_inference.constants import SUPPORTED_POLICIES; print(SUPPORTED_POLICIES)'
```

### `model.safetensors not found`

이 패치의 `UniVLAPolicy.from_pretrained()`는 dummy artifact에서는 `model.safetensors` 없이 config만으로 로드되도록 override되어 있습니다.

그래도 이 에러가 나면 원인 후보는 둘 중 하나입니다.

- 실제로 패치된 `UniVLAPolicy`가 아니라 기본 `PreTrainedPolicy.from_pretrained()`가 호출되고 있음
- `policy_type`이 `univla`가 아니라 다른 policy로 들어가고 있음

확인:

```bash
python -c 'from lerobot.policies.factory import get_policy_class; print(get_policy_class("univla").from_pretrained)'
```
