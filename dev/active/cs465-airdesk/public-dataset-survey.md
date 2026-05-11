# AirDesk Public Dynamic Gesture Dataset Survey

Date: 2026-05-10

## Decision

Start with **IPN Hand**, then optionally use **IPN HandS** annotations if the
released files are available locally. IPN is the best first public-data fit
because it is continuous, RGB-webcam based, explicitly includes natural
non-gesture hand motion, and has touchless-screen gesture classes that map
directly to AirDesk atomic left/right swipe evidence.

Jester is valuable later for large clip-level pretraining, but it is not the
best first dataset for AirDesk's current problem. It is made of short isolated
clips, so it helps class separability more than boundary spotting, natural
background rejection, and repeated-event decoding.

## Architecture Fit

Keep the current AirDesk recognizer contract:

- train atomic evidence heads: `intentional_motion`, `stroke_left`,
  `stroke_right`, `start`, and `end`;
- do not train combo classes such as `right_right_left`;
- interpret combos in a later command-grammar layer over emitted atomic events;
- keep public data as a training aid, not the authority for AirDesk success.

For the first IPN pass, map only IPN `G05 Throw left` / `G06 Throw right` into
AirDesk's existing left/right atomic evidence labels. This is a proxy for
lateral motion evidence, not a claim that IPN contains AirDesk swipe gestures.
Keep IPN click, double-click, zoom, point, open, and non-gesture intervals as
background/negative evidence until AirDesk adds explicit heads for click/select,
push, or other atomic gestures.

## Dataset Comparison

| Dataset | License / access | Classes | Continuous or clip-level | Subjects / samples | Modality | AirDesk mapping |
|---|---|---:|---|---|---|---|
| IPN Hand | Data and annotations are CC BY 4.0 on the official project page; MP4 download is about 4.6 GB and annotations are about 364 KB. | 13 gesture classes plus non-gesture labeling table entries | Continuous videos; each subject performed multiple gestures with random breaks in one video | 50 subjects; over 4,000 gesture instances and about 800,000 RGB frames; official table lists 200 videos | RGB video, 640x480, 30 FPS; optional optical-flow and segmentation derivatives | Best first fit. `G05 Throw left` -> `swipe_left`; `G06 Throw right` -> `swipe_right`; `G01/G02` are possible future click/select; non-gesture is directly useful negative data. |
| IPN HandS | Planned/released as CC BY 4.0 derived annotations over IPN Hand; verify local availability before relying on it. | Refined 14-class taxonomy | Continuous videos with frame-level skeleton annotations and revised temporal boundaries | Paper reports 7770 instances, roughly 800,000 frames, over 700k hand skeletons | RGB-derived 21-landmark hand skeletons | Potentially better than automatic MediaPipe for skeleton training, but use only after confirming the files are actually accessible. |
| Jester / 20BN-JESTER | Original 20BN site is referenced by the paper but current primary access is uncertain; mirrors exist, but license should not be assumed from mirrors. | 27 classes: 25 gesture classes plus `No gesture` and `Doing other things` | Short clip-level classification, about 3 seconds per clip | 148,092 clips, 5,331,312 frames, 1,376 actors | RGB frame bursts at 12 FPS, height 100 px and variable width | Good later for large-scale clip pretraining. Has swipe, push, pull, zoom, no-gesture, and doing-other-things classes, but lacks continuous interval boundaries. |
| EgoGesture | Requires signed agreement and approval; research/educational use with rights reserved. | 83 static/dynamic gestures | Supports both segmented classification and continuous detection; sessions contain 9-14 random gestures | 50 subjects; 2,081 RGB-D videos; 24,161 samples; 2,953,224 frames | Egocentric RGB-D from RealSense SR300 at 640x480, 30 FPS | Technically strong for continuous detection, but viewpoint is head-mounted/wearable rather than desktop webcam, so transfer to AirDesk may be weird. |
| ChaLearn LAP ConGD | Challenge dataset from ChaLearn/CodaLab lineage; access is less simple than IPN and class vocabulary is broad. | 249 labels | Continuous gesture spotting/recognition | 21 subjects; 47,933 gestures in 22,535 RGB-D videos | RGB-D | Useful benchmark shape, but gesture vocabulary spans sign/domain gestures rather than AirDesk-like desktop controls. Higher setup cost. |
| NVGesture | NVIDIA page links dataset; paper page is available, but dataset license/access should be checked before download. | 25 classes | Unsegmented multimodal input streams for online detection/classification | 1,532 dynamic gestures in commonly reported split; paper focus is automotive control | RGB, depth, stereo-IR | Useful later for multimodal dynamic gesture baselines, less aligned than IPN because it is smaller and automotive-context shaped. |
| HaGRIDv2 | Public research dataset; useful for static/dynamic images and negative/no-gesture diversity. | Many image classes; no-gesture diversified | Primarily image/static or image-derived gesture recognition, not continuous video spotting | About 1M images in v2 paper | RGB images | Not a first TCN event-source. Could help later for static hand shape and false-positive rejection, not boundary training. |

## Source Notes

- IPN official page: https://gibranbenitez.github.io/IPN_Hand/
- IPN code/annotations repository: https://github.com/GibranBenitez/IPN-hand
- IPN Hand paper: https://arxiv.org/abs/2005.02134
- IPN HandS paper: https://www.mdpi.com/2076-3417/15/11/6321
- Jester paper: https://openaccess.thecvf.com/content_ICCVW_2019/html/HANDS/Materzynska_The_Jester_Dataset_A_Large-Scale_Video_Dataset_of_Human_Gestures_ICCVW_2019_paper.html
- EgoGesture official page: https://nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html
- ChaLearn LAP paper: https://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w18/html/Wan_ChaLearn_Looking_at_CVPR_2016_paper.html
- NVIDIA dynamic hand gesture page: https://research.nvidia.com/publication/2016-06_online-detection-and-classification-dynamic-hand-gestures-recurrent-3d
- HaGRIDv2 paper: https://arxiv.org/abs/2412.01508

## Implemented Importer

The first importer is intentionally small and local-data-only:

```bash
uv run airdesk public-data ipn-convert \
  --videos-dir data/public/ipn/videos \
  --annotations-dir data/public/ipn/annotations-download \
  --out-dir data/public/ipn/airdesk \
  --split train \
  --limit 1 \
  --manifest-out data/public/ipn/airdesk/tcn-v2-ipn-smoke-manifest.json \
  --mapping-out data/public/ipn/airdesk/ipn-airdesk-mapping.csv
```

As of 2026-05-10, the official IPN Hand Drive annotations and all five video
archives have been downloaded into ignored local storage under
`data/public/ipn/`; extraction produced 200 `.avi` files under
`data/public/ipn/videos/`. A one-video, 120-frame smoke conversion succeeded
against the official annotation filenames.

Outputs:

- `recordings/*.jsonl`: AirDesk replay recordings produced by running IPN MP4
  video through MediaPipe Hand Landmarker.
- `labels/*.labels.json`: AirDesk label files with only `G05` / `G06` mapped as
  positive swipe events/phases.
- `features/*.csv`: normal AirDesk `FrameFeatureRow` CSVs.
- optional manifest: `stream-invariant-v2`, `target-mode=v2-evidence`.

Raw IPN downloads should stay under ignored `data/public/` or another ignored
local path. Do not commit public dataset videos, extracted frames, generated
features, labels, or model checkpoints unless Caden explicitly asks for a tiny
fixture.

## Next Experimental Steps

1. Caden downloads the IPN MP4 videos and annotations locally.
2. Convert a one-video smoke with `--limit 1 --frame-limit 120` to verify the
   MediaPipe path and output shape quickly.
3. Convert the full IPN train/validation splits into ignored `data/public/ipn/`.
4. Build an IPN-only `stream-invariant-v2` / `v2-evidence` manifest.
5. Train an IPN-only schema-2 TCN v2 on the `G05` / `G06` throw-left/right proxy
   mapping and evaluate replay feel.
6. Compare AirDesk-only, IPN-only, IPN-pretrain/AirDesk-fine-tune, and hybrid
   training on the AirDesk source-held-out `data/recordings/v2-*` recordings.

## First IPN-Only TCN V2 Result

The first IPN-only model was trained from
`data/public/ipn/airdesk-train/tcn-v2-ipn-train-manifest.json`; no AirDesk local
recordings were mixed into this checkpoint. The model path is
`data/models/gestures/tcn-v2-ipn-train-atomic-10ep.pt`.

Training used the full converted IPN train split: 148 source videos, 4,039 IPN
segments, 296 mapped `G05` / `G06` left/right throw segments, and 99,510 TCN v2
windows. The held-out converted IPN test split contains 52 source videos, 1,610
IPN segments, 104 mapped left/right throw segments, and 35,706 windows.

Evaluation confirms the proxy is useful but noisy:

- low decoder thresholds (`activation=0.35`, `release=0.2`, `min_peak=0.35`)
  matched `99/104` held-out throw-left/right events, but produced `2,183` false
  activations from `2,374` decoded candidates;
- default thresholds matched `91/104`, with `915` false activations;
- stricter `0.80/0.45/0.80` thresholds matched `88/104`, with `247` false
  activations;
- very strict `0.90/0.50/0.90` thresholds matched `52/104`, with `49` false
  activations.

Interpretation: the IPN-only model learns strong left/right lateral motion
evidence, but treating all other dynamic IPN gesture classes as plain background
makes the current two-head decoder fire on lots of non-left/right gesture
motion. Do not treat this as an AirDesk swipe model. The next useful experiment
is either better public-data target design, for example an explicit
`intentional_non_left_right_motion` / richer atomic heads, or IPN pretraining
followed by AirDesk fine-tuning and evaluation on AirDesk source-held-out V2
recordings.

## All-IPN Gesture Pivot

Caden correctly flagged that the first two-head IPN proxy model should not be
judged as broadly bad from false activations on other IPN gestures. Those
"false" activations were partly caused by the target design: the model was asked
to learn only `G05` / `G06`, while the other 11 non-background IPN gesture
classes were treated as background.

The next public-data target is therefore an IPN-only all-gesture evidence model.
AirDesk now supports custom v2 evidence heads, and IPN can generate an
`ipn-all` label mode with:

- `intentional_motion`;
- one head for every non-`D0X` IPN class:
  `ipn_b0a`, `ipn_b0b`, `ipn_g01` ... `ipn_g11`;
- `start` and `end` boundary heads.

Ignored all-IPN manifests have been generated from the existing tracked IPN
recordings without rerunning MediaPipe:

- train: `data/public/ipn/airdesk-train-ipn-all/tcn-v2-ipn-all-train-manifest.json`
  with 148 sources, 3,117 labeled IPN gesture events, and 99,510 windows;
- held-out test:
  `data/public/ipn/airdesk-test-ipn-all/tcn-v2-ipn-all-test-manifest.json`
  with 52 sources, 1,101 labeled IPN gesture events, and 35,706 windows.
