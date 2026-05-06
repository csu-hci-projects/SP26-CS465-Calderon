# Related Work Synthesis

This is the working version of the related-work argument. It can be promoted into
`paper/latex-source/main.tex` once the pilot evidence is final.

## 1. Situational Impairment and Ability-Based Desktop Control

AirDesk should open related work by grounding the problem in situationally induced
impairments and disabilities (SIIDs). SIIDs describe contexts where a user's
ability to interact is reduced by environment or situation rather than by a
permanent impairment alone. The strongest existing SIID literature is mobile-first,
but the concept transfers well to desktop work: dirty hands, gloves, painting,
cooking, hardware repair, standing away from the desk, and wrist strain can all
make keyboard/mouse input temporarily costly.

Ability-based design gives the more careful accessibility frame. The key move is
to focus on available abilities and system adaptation. For AirDesk, that means
gestures should be offered as an additional mode when they fit the user's current
body/context, not as a universal replacement. This is also where we keep our
claims honest: AirDesk is accessibility-motivated and ability-aware, but not
validated as an accessibility tool for people with arthritis, RSI, tremor, or other
specific conditions unless those users are actually studied.

Best citation cluster: `\cite{sarsenbayeva2019siids,wobbrock2011ability}`.

## 2. Mid-Air Interaction Is Context-Specific

The mid-air interaction literature repeatedly pushes against universal gesture
sets. The Vogiatzidakis and Koutsabasis elicitation review is especially useful:
it says mid-air gesture identification depends on context of use and that there is
no established universal vocabulary. Catalano and Luo's recent usability review
adds that usability outcomes vary by evaluation method, application, gesture
design, testing method, and participant factors such as familiarity, training, and
physical ability. Aigner et al. add a complementary point: people's preferred
gesture types vary by intended meaning, so gesture designers need to choose from
pointing, direct manipulation, semaphoric, iconic, and other gesture types
according to task semantics.

For AirDesk, this supports a small vocabulary designed for desktop commands:
open palm to clutch/listen, directional swipes for workspace navigation, pointing
for focus movement, fist/cancel for exit, and pinch only in separate modes. The
claim should be that this vocabulary is plausible and testable for AirDesk's
desktop-command setting, not that it is universally optimal.

Best citation cluster:
`\cite{koutsabasis2019empiricalmidair,catalano2025usablewithouttouch,vogiatzidakis2018gestureelicitation,aigner2012midairgestures,wittorf2016wallgestures}`.

## 3. Why AirDesk Should Not Claim to Replace Keyboard and Mouse

The strongest counterweight is Jakobsen et al.'s touch-vs-mid-air comparison.
Their large-display study found that mid-air can be slower than touch, but people
choose mid-air more when the physical movement cost of touch increases. This is
basically AirDesk's thesis in a nearby domain: ordinary input is excellent when
available and convenient; mid-air becomes interesting when ordinary input has a
contextual cost.

That evidence lets the paper be confident without overclaiming. AirDesk can say
it targets coarse, low-text, secondary commands during situational impairment.
It should avoid saying gestures are faster than shortcuts, better for precision,
or more productive for normal seated desktop use.

Best citation cluster: `\cite{jakobsen2015touchmidair}`.

## 4. Fatigue, False Activations, and the Need for Intent Gating

Mid-air interaction has real interaction costs. Hincapie-Ramos et al. show that
arm fatigue is a measurable design problem. Arif et al. show that error-prone
in-air recognition changes user behavior and that errors can come from both human
performance and the recognizer/system. Vogiatzidakis and Koutsabasis add a useful
design analogy: their smart-home prototype required a registration gesture before
command gestures so only the active target could respond. Together, these sources
justify AirDesk's main design constraints: small command vocabulary, short
movements, explicit clutching, dry-run defaults, feedback, cooldowns,
false-activation metrics, and separation of command mode from cursor/text modes.

This is also the right place to discuss "Midas touch" in plain language: when a
system watches continuously, ordinary motion can be mistaken for commands. AirDesk
should describe this as false activation / unintended activation unless we add a
specific Midas-touch citation.

Best citation cluster:
`\cite{hincapie2014consumedendurance,arif2014errorprone,vogiatzidakis2020homedevices,walter2014cuenesics}`.

## 5. Recognition: From Rule Scaffolds to Templates to Temporal Models

AirDesk should not turn the paper into a model-comparison paper. The recognizer
story is narrower: rules are inspectable but brittle; template/DTW matching is a
reasonable low-data prototype baseline; temporal models are future work once the
project has enough continuous labeled data.

Wobbrock et al.'s $1 recognizer paper is useful because it legitimizes simple,
low-data recognizers for UI prototypes and explicitly compares simple template
recognition to DTW and Rubine. Lea et al.'s TCN paper supports the future direction
of temporal segmentation/detection for continuous streams. MediaPipe Hands should
be cited only as implementation background for real-time RGB hand landmarks.

Best citation cluster:
`\cite{wobbrock2007dollar,lea2017tcn,zhang2020mediapipehands}`.

## Draft Paragraph Shape

The final related-work section can be four subsections:

1. Situational and ability-based input.
2. Mid-air gesture design.
3. Interaction costs: precision, fatigue, errors, and intent.
4. Recognition and implementation background.

That shape keeps the section tight and connected to AirDesk's system design.
