# Tongue ROM Dataset Collection — User Guide

## What this is

A short browser app that records 3 tongue movement tasks for each participant and stores the clips with a clinical score. It runs in any modern browser (Chrome / Edge / Safari, desktop or mobile).

## Why we are collecting data

We are building a small labeled dataset so we can train a convolutional neural network (CNN) to **predict tongue range-of-motion scores automatically** from short video clips. Manual clinical scoring (Lazarus et al., 2014 rubric) is time-consuming and subjective; an automated scorer would let clinicians triage oral-cancer patients faster and more consistently.

To train the CNN we need many examples across the **full range of impairment** — not just healthy "score 100" performances. That is why we ask participants to deliberately **simulate different levels of impairment**, including moderate and severe cases, even if their tongue is perfectly healthy.

## Privacy & data handling

- **Stored securely on an encrypted lab machine.** Raw clips never leave the lab network.
- **Not redistributed.** The dataset is used only inside our research group to train the model.
- **Anonymized before any downstream use.** After collection each clip is cropped so that only the **buccal / mouth region** is retained — facial identity is removed. The "name" field is used only to group the 3 clips of a single session on disk and is not part of the final dataset.

## How the app works

1. **Enter your name.** Any name is fine (letters, digits, spaces, `_ . -`, up to 64 chars).
2. **Task instructions screen.** You will see which of the 3 tasks is next (lateralization right, lateralization left, elevation) and how to perform it. Press *I'm ready*.
3. **Live capture.** The camera turns on. A green face mesh is drawn on top of your face. When you hold the requested tongue pose, a progress bar fills from 0 s to 2 s. Once the 2 s hold is complete, the camera automatically stops and the last ~2 seconds of video are kept.
4. **Scoring.** The saved clip plays back. You (or the clinician) pick the score that matches the pose actually performed in the clip. The score is uploaded together with the clip and the face-mesh landmarks.
5. **Repeat** for the other 2 tasks. When all 3 are done you can start a new participant.

## Please simulate the full range — do NOT always aim for 100

The model cannot learn what "impaired" looks like if every clip we feed it is a healthy "100". To make the dataset useful, each participant should perform **a mix of deliberate impairment levels across the session**, for example:

- A clip that would clinically score **100** — full healthy movement.
- A clip that would clinically score **50** — deliberately hold back, only partway to the target.
- A clip that would clinically score **0** — barely move the tongue at all, or stop at the starting position.

Mix them up across tasks and repeats. After each capture, **honestly label the clip with the score the pose actually deserves** — not the score you were trying to reach. The label is what matters: a clip where you held at 50% should be labeled `50`, even if you meant to do 100 but slipped.

## The clinical rubric (Lazarus et al., 2014)

The app uses the Lazarus tongue ROM rubric. Scores in the app are the ones you will see as buttons.

### Lateralization — Right *and* Left

Move the tongue toward the corresponding corner of the mouth (the subject's own right / own left). Hold the maximum position.

| Score | Clinical label          | Definition                                                             |
|-------|-------------------------|------------------------------------------------------------------------|
| 100   | Normal                  | Tongue touches the corner of the mouth (commissure)                    |
| 50    | Mild–moderately impaired| **< 50 %** reduction in tongue movement from the corner of the mouth   |
| 25    | Severely impaired       | **> 50 %** reduction in tongue movement from the corner of the mouth   |
| 0     | Totally impaired        | No tongue movement in either direction                                 |

### Elevation

Open the mouth and lift the tongue tip up to touch the roof of the mouth, just behind the upper teeth. Hold the maximum position.

| Score | Clinical label         | Definition                                                         |
|-------|------------------------|--------------------------------------------------------------------|
| 100   | Normal                 | Tongue tip contacts the upper alveolar ridge                       |
| 50    | Moderately impaired    | Tongue tip elevation is visible but **no contact** with the ridge  |
| 0     | Severely impaired      | No visible tongue tip elevation                                    |

## Tips for a good capture

- **Good lighting on the face**, especially around the mouth. HSV-based tongue detection fails in dim light.
- **Face fills most of the frame** and is roughly centered. If no face is detected the app will prompt you.
- **Keep the phone / webcam still** during the 2-second hold.
- If the auto-hold detector cannot trigger (e.g. for a "score 0" clip where the tongue does not move), press **Capture now** to save the clip manually.

## If something goes wrong

- *Camera permission denied* — reload the page and allow camera access.
- *"Tongue not visible"* — open the mouth wider and check lighting.
- *Upload failed* — click the score button again to retry; the clip is held in memory until upload succeeds.

---

*Sanchez Research Lab, UIC ECE — Tongue ROM data collection for CNN training.*
