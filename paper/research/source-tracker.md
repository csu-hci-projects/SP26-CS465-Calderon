# Source Tracker

Use this file to track source status before and after sources are promoted into
`latex-source/references.bib`.

## Claim Buckets

- Situational impairment and ability framing.
- Mid-air interaction design and gesture vocabulary.
- Touchless interaction trade-offs, fatigue, and false activation.
- Gesture recognition baselines and continuous temporal recognition.
- Hand tracking implementation background.
- Desktop/window-manager implementation references.

## Core Sources Already Downloaded

| Status | Source | Bucket | Why It Matters | Local PDF |
| --- | --- | --- | --- | --- |
| Use | Wobbrock et al. 2011, "Ability-Based Design" | Ability framing | Supports the argument that systems should adapt to users' available abilities and context. Use carefully: AirDesk is accessibility-motivated, not accessibility-validated. | `paper/PDFs-LiteratureSurvey/wobbrock2011_ability_based_design.pdf` |
| Use | Sarsenbayeva et al. 2019, "Situationally-Induced Impairments and Disabilities Research" | SIIDs | Establishes SIIDs as an HCI frame and motivates temporary/contextual barriers to ordinary input. Mostly mobile-focused, so AirDesk extends the frame to desktop control. | `paper/PDFs-LiteratureSurvey/sarsenbayeva2019_siids_research.pdf` |
| Use | Koutsabasis and Vogiatzidakis 2019, "Empirical Research in Mid-Air Interaction: A Systematic Review" | State of the art | Broadest state-of-the-art anchor: 104 empirical papers, application domains, prototyping/design issues, evaluation methods, and challenges. | Manual full-text recommended; abstract and metadata found online |
| Use | Vogiatzidakis and Koutsabasis 2018, "Gesture Elicitation Studies for Mid-Air Interaction: A Review" | Gesture design | Corrects our earlier seed citation. Useful for arguing there is no universal mid-air gesture vocabulary and that gesture choice is context-specific. | `paper/PDFs-LiteratureSurvey/vogiatzidakis2018_midair_elicitation_review.pdf` |
| Use | Catalano and Luo 2025, "Usable Without Touch? Revealing the Relationship Between Usability and Mid-Air Gestures" | Current usability review | Recent systematic review of 10 studies. Useful for context-aware design, feedback modality, training, participant factors, and standardized usability evaluation gaps. | `paper/PDFs-LiteratureSurvey/catalano2025_usable_without_touch.pdf` |
| Use | Aigner et al. 2012, "Understanding Mid-Air Hand Gestures" | Gesture vocabulary | Human preference study of 5,500 gestures from 12 participants; useful for gesture types, pointing, direct manipulation, semaphoric/iconic gestures, and context-dependent mappings. | `paper/PDFs-LiteratureSurvey/aigner2012_understanding_midair_hand_gestures.pdf` |
| Use | Jakobsen et al. 2015, "Should I Stay or Should I Go?" | Touch vs mid-air trade-offs | Good counterweight against overclaiming: mid-air is often slower than touch, but users choose it when the physical cost of touch rises. This maps directly to AirDesk's "secondary channel" thesis. | `paper/PDFs-LiteratureSurvey/jakobsen2015_touch_vs_midair_large_display.pdf` |
| Use | Hincapie-Ramos et al. 2014, "Consumed Endurance" | Fatigue | Establishes arm fatigue as a real design/evaluation concern for mid-air interaction; justifies short gestures, low repetition, and fatigue notes in the pilot. | `paper/PDFs-LiteratureSurvey/hincapie2014_consumed_endurance.pdf` |
| Use | Walter et al. 2014, "Cuenesics" | Public-display mid-air selection | Field evaluation of mid-air selection techniques; useful for gesture phases, discoverability, immediate use, and selection in non-desktop touchless contexts. | `paper/PDFs-LiteratureSurvey/walter2014_cuenesics.pdf` |
| Use | Wittorf and Jakobsen 2016, "Eliciting Mid-Air Gestures for Wall-Display Interaction" | Gesture elicitation | 20 participants, 25 actions, 1124 gestures. Useful evidence that gesture sets depend on display/task context and often become large/physical for wall displays. | `paper/PDFs-LiteratureSurvey/wittorf2016_midair_wall_display.pdf` |
| Use | Vogiatzidakis and Koutsabasis 2020, "Mid-Air Gesture Control of Multiple Home Devices in Spatial Augmented Reality Prototype" | Intent gating / Midas touch | Strong analogy for AirDesk's clutch/listen pattern: users register a device before command gestures, reducing accidental activation across many command targets. | `paper/PDFs-LiteratureSurvey/vogiatzidakis2020_midair_home_devices_registration.pdf` |
| Use | Arif et al. 2014/2021, "How Do Users Interact with an Error-Prone In-Air Gesture Recognizer?" | Error behavior | Strong support for AirDesk's safety-first stance: in-air gestures are error-prone in practice, and feedback/errors/fatigue shape use. | `paper/PDFs-LiteratureSurvey/arif2014_error_prone_in_air_gesture_recognizer.pdf` |
| Use | Wobbrock et al. 2007, "$1 Recognizer" | Template recognition | Gives language for low-data template recognizers and comparisons to DTW/Rubine. Useful for explaining why AirDesk uses DTW as a practical baseline before learned models. | `paper/PDFs-LiteratureSurvey/wobbrock2007_dollar_recognizer.pdf` |
| Use | Lea et al. 2017, "Temporal Convolutional Networks for Action Segmentation and Detection" | Temporal recognition | Supports future causal temporal-model direction for continuous gesture spotting/segmentation over feature streams. | `paper/PDFs-LiteratureSurvey/lea2017_tcn_action_segmentation.pdf` |
| Use | Zhang et al. 2020, "MediaPipe Hands" | Hand tracking | Implementation background only: MediaPipe provides real-time RGB hand landmarks; AirDesk contribution is interaction/system design, not hand tracking. | `paper/PDFs-LiteratureSurvey/zhang2020_mediapipe_hands.pdf` |
| Use | Hyprland Dispatchers docs | Desktop implementation | Use sparingly for implementation target; probably belongs in system overview or reproducibility rather than related work. | n/a |

## Manual Grab / Maybe Sources

| Priority | Source | Why It Might Matter | Current Status |
| --- | --- | --- | --- |
| High | Koutsabasis and Vogiatzidakis 2019, "Empirical Research in Mid-Air Interaction: A Systematic Review" | This is the best broad state-of-the-art paper for the related-work section. | Full text appears paywalled at Taylor & Francis. I found abstract, metadata, and ResearchGate snippets, but not a clean open PDF. |
| Medium | Chen et al. 2026, "PATE Model: A 30-Year Review and Analysis of Gestural Interaction Research" | Very current broad gestural-interaction review. Could help if we want a newer state-of-the-art sentence. | Abstract/metadata visible; likely paywalled. |
| Low | Rubine 1991, "Specifying Gestures by Example" | Classic gesture-recognition citation. Useful if we need historical depth, but Wobbrock 2007 already covers AirDesk's low-data recognizer story better. | Search found ResearchGate/CiteSeer copies; not necessary yet. |
| Medium | "Continuous hand gesture recognition: Benchmarks and methods" 2025 | Would help if the final recognition section needs current continuous-gesture spotting language and false-positive benchmarks. | Abstract found; full text appears paywalled. |
| Low | Dynamic gesture spotting papers beyond TCNs | Could deepen the recognition section if the final paper spends more space on models. | Defer unless recognition becomes a larger contribution. |

## Working Notes

The related work should not be a literature dump. Each source should support one
of these AirDesk claims:

- Situational impairment is a defensible HCI frame, but current AirDesk evidence is
  formative and desktop-specific.
- Mid-air gesture vocabularies are context-dependent; AirDesk should justify a
  small desktop-command vocabulary rather than imply universality.
- Mid-air input has costs: fatigue, ambiguity, false activations, discoverability,
  and weaker precision than direct touch/keyboard/mouse.
- Those costs make clutching, dry-run mode, feedback, replay, false-positive
  measurement, and limited command scope central design choices.
- Template/DTW recognition is a practical low-data baseline, while learned temporal
  models are future work until more continuous labeled data exists.
