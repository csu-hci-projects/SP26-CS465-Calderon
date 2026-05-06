# Literature Matrix

This matrix turns sources into claims AirDesk can actually use. The goal is to
write related work as an argument, not as a bibliography tour.

| Source | Bucket | Key Claim / Finding | How AirDesk Uses It | Citation Key | PDF Status |
| --- | --- | --- | --- | --- | --- |
| Wobbrock et al. 2011, "Ability-Based Design" | Ability/accessibility framing | Accessible systems should focus on what users can do and adapt systems to user abilities and contexts. | Positions AirDesk as an optional input layer that adapts desktop control to temporary ability/context constraints. Do not claim AirDesk is validated for disabled users. | `wobbrock2011ability` | Downloaded |
| Sarsenbayeva et al. 2019, "Situationally-Induced Impairments and Disabilities Research" | SIIDs | Environmental/contextual factors can impair interaction even for users without permanent disabilities; SIIDs need systematic study. | Motivates dirty hands, gloves, distance from keyboard, pain/fatigue, and occupied-hands scenarios as legitimate HCI problems. | `sarsenbayeva2019siids` | Downloaded |
| Koutsabasis and Vogiatzidakis 2019, "Empirical Research in Mid-Air Interaction" | State of the art | Mid-air interaction is a distinct HCI style; review covers domains, prototyping/design issues, and empirical evaluation methods across 104 publications. | Sets the state-of-the-art frame and helps justify why AirDesk must discuss evaluation, prototyping maturity, feedback, and domain specificity. | `koutsabasis2019empiricalmidair` | Manual full text recommended |
| Catalano and Luo 2025, "Usable Without Touch?" | Current usability review | A systematic review of 10 mid-air gesture usability studies found usability varies by evaluation method, application, gesture design, testing method, prior experience, training, and physical ability. | Fresh support for AirDesk's context-aware design, feedback, training, participant-factor notes, and honest pilot limitations. | `catalano2025usablewithouttouch` | Downloaded |
| Vogiatzidakis and Koutsabasis 2018, "Gesture Elicitation Studies for Mid-Air Interaction" | Gesture vocabulary | Mid-air interaction lacks an established universal vocabulary; gesture identification depends on context of use. | Supports AirDesk's narrow vocabulary and lets us argue that workspace/focus/media commands need their own design rationale. | `vogiatzidakis2018gestureelicitation` | Downloaded |
| Aigner et al. 2012, "Understanding Mid-Air Hand Gestures" | Gesture types | In a 12-participant study of 5,500 gestures, preferred gesture types varied by meaning/action. | Supports choosing direct, semaphoric, and pointing-like primitives according to command semantics rather than copying mouse input. | `aigner2012midairgestures` | Downloaded |
| Wittorf and Jakobsen 2016, "Eliciting Mid-Air Gestures for Wall-Display Interaction" | Gesture elicitation | Wall-display gestures were influenced by surface interaction and tended to be larger/physical in that context. | Shows task/display context changes gesture choice; AirDesk's desktop-scale webcam setup should avoid oversized wall-display-style gestures. | `wittorf2016wallgestures` | Downloaded |
| Jakobsen et al. 2015, "Should I Stay or Should I Go?" | Touch vs mid-air | Mid-air was generally slower than touch for target acquisition, but users chose mid-air when the movement cost of touch increased. | Central to AirDesk thesis: gestures are not a replacement for keyboard/mouse; they are useful when the cost of ordinary input rises. | `jakobsen2015touchmidair` | Downloaded |
| Hincapie-Ramos et al. 2014, "Consumed Endurance" | Fatigue | Mid-air interactions are prone to upper-limb fatigue; fatigue can be quantified and should inform design. | Justifies short command gestures, low-repetition tasks, fatigue measures, and not evaluating long text-entry/cursor sessions as the main contribution. | `hincapie2014consumedendurance` | Downloaded |
| Walter et al. 2014, "Cuenesics" | Selection/discoverability | Mid-air selection techniques for public displays need immediate usability and clear phases for selection. | Useful analogy for AirDesk mode phases: intent/clutch, command execution, confirmation/cancel, cooldown. | `walter2014cuenesics` | Downloaded |
| Vogiatzidakis and Koutsabasis 2020, "Mid-Air Gesture Control of Multiple Home Devices" | Intent gating / accidental activation | The system uses registration gestures before command gestures and explicitly discusses this as a forcing function against the Midas touch problem. | Direct support for AirDesk's open-palm clutch/listen mode and command/cooldown policy. | `vogiatzidakis2020homedevices` | Downloaded |
| Arif et al. 2014/2021, "How Do Users Interact with an Error-Prone In-Air Gesture Recognizer?" | Recognition errors | In-air gesture interfaces are more error-prone in practice due to ambiguity and lack of feedback; the paper separates human and system errors. | Supports measuring false activations, missed gestures, repeated fires, and perceived control instead of reporting only offline accuracy. | `arif2014errorprone` | Downloaded |
| Wobbrock et al. 2007, "$1 Recognizer" | Template recognition | Simple template recognizers can perform well for UI prototypes with few examples; the study compares $1 with DTW and Rubine. | Frames AirDesk's DTW baseline as an appropriate low-data prototype recognizer, not a final model. | `wobbrock2007dollar` | Downloaded |
| Lea et al. 2017, "Temporal Convolutional Networks" | Temporal models | TCNs model temporal action segmentation/detection with hierarchical temporal convolutions and can be faster to train than LSTMs in their setting. | Supports future work on causal temporal recognition over continuous AirDesk feature streams after more labeled data exists. | `lea2017tcn` | Downloaded |
| Zhang et al. 2020, "MediaPipe Hands" | Hand tracking | Real-time hand skeleton tracking from RGB camera using palm detection plus landmark model. | Implementation background for AirDesk's first tracking backend; keep contribution focused on desktop-control system design. | `zhang2020mediapipehands` | Downloaded |
| Hyprland Dispatchers docs | Implementation | Hyprland exposes dispatcher commands for window/workspace control. | Supports system reproducibility and action-routing details. | `hyprlanddispatchers` | Web |

## Synthesis Claims To Carry Into Related Work

1. AirDesk is best positioned as a secondary command channel under situational
   impairment, not a faster or universally better desktop input method.
2. Gesture design must be context-specific. The literature does not support a
   universal vocabulary, so AirDesk should explain why its initial vocabulary is
   small, coarse, and desktop-command oriented.
3. Safety and intent gating are not implementation polish; they are core HCI
   requirements because in-air gesture input is ambiguous, error-prone, and can
   fatigue users.
4. The technical recognizer story should stay modest: rule recognizers are useful
   as scaffolding, DTW/template methods are credible low-data baselines, and TCNs
   are a future learned model once there is enough continuous labeled data.

## Evidence Gaps

- We still need one or two strong sources on continuous gesture spotting/false
  positive rejection if the recognition section grows beyond a page.
- The Koutsabasis and Vogiatzidakis 2019 systematic review should be manually
  grabbed if possible because it is probably the best state-of-the-art anchor.
- If the instructor expects even more recent literature, manually grab Chen et al.
  2026 or the 2025 continuous hand gesture recognition benchmark paper; they look
  useful but are not essential for the first AirDesk draft.
