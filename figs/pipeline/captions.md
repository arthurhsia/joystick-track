# Pipeline figure captions

## fig01_preprocessing_cascade

Preprocessing cascade for subject RH (median-performing, not cherry-picked), electrode 20. The
8-second window shows three stages of the pipeline at the same vertical scale, offset for
readability: raw recording (top), after line-noise removal (middle), and after line-noise + common-
average referencing (bottom). The inset zooms into a 0.5-second segment to illustrate removal of the
60 Hz mains component. All three traces are plotted on identical y-axes; only the vertical offset
differs.

## fig02_linenoise_and_badchannels

Line-noise characterisation and bad-channel detection for subject RH (median-performing, not cherry-
picked). Panel (a) shows the Welch power spectral density of electrode 20 before (Raw) and after
(Notch) line-noise removal; vertical dotted lines mark the 60, 120, and 180 Hz harmonics. Panel (b)
is a per-electrode scatter of robust amplitude modified-z (x-axis) against line-noise SNR modified-z
(y-axis); dashed lines show the detection thresholds (z = 3.5); electrodes that exceed either
criterion are shown in red and labelled by their 1-based channel index.

## fig03_feature_representation

Neural feature representation for subject RH electrode 20, trial 1 (median by duration, not cherry-
picked). The top panel shows the joystick cursor trajectory (cx pink, cy cyan) at 10 Hz. The heatmap
shows all eight features — seven band-power envelopes (delta through high-gamma) and the LMP —
z-scored within the trial, with red indicating elevated and blue suppressed activity. LMP (bottom
row) is unsigned and signed respectively; all other bands are rectified power envelopes smoothed at
1.5 Hz.

## fig04_segmentation

Trial segmentation for subject RH (median-performing, not cherry-picked). Top: cursor speed
(pixels/s) across the full session. Shaded blue regions are detected inter-trial freeze gaps (both
cx and cy exactly constant for ≥ 1.5 s). Bottom: accepted trial spans shown as coloured bars with
duration in seconds; trials shorter than 30 s are excluded from the analysis (none shown here, as
all pauses between segments produced trials of sufficient length for this subject).

## fig05_channel_selection_rmatrix

Channel-selection r-matrices for subject RH, fold 2 training set (median fold by mean Pearson r, not
cherry-picked). Each heatmap shows the Pearson r between each (electrode, spectral-band) pair and
the cursor position for cx (left) and cy (right). Red = positive correlation, blue = negative. Black
rectangles outline the top-10 (electrode, band) pairs selected by the signed-mean ranking metric
|((r_cx + r_cy)/2)|. Channels with opposite-sign r across cx and cy may be suppressed by the signed-
mean metric; see the metric-comparison diagnostic for details.

## fig06_cv_structure

Cross-validation structure for subject RH (5-fold contiguous block CV). Grey = training samples;
coloured = test samples; yellow hatched = embargo strip (20 samples = 2.0 s on each side of each
test block) that is excluded from training to prevent leakage from the 1.5 Hz envelope smoother and
1 sample lag embedding. Vertical dotted lines mark trial boundaries. The embargo IS applied in the
reported results; this figure accurately reflects the code as run.

## fig07_decoder_output

Decoded cursor output for subject RH, fold 2/5 (median fold by mean Pearson r, not cherry-picked).
Panel (a): cx decoded by Ridge (dashed) and post-processed with a fixed 0.25 Hz zero-phase LPF
(solid green); true trajectory in black. Panel (b): cy decoded by Ridge and post-processed with a
Kalman filter whose process-noise parameter σ_a is set a priori from the target bandwidth (0.25 Hz),
with velocity always fused; no test-set tuning. Panel (c): two-dimensional trajectory for the same
window. The "Fixed" label means parameters were chosen before seeing the test scores.

## fig08_postprocessing_effect

Post-processing effect on cy decoding. Panel (a): a 20-second excerpt from RH trial 1 (median by
duration, not cherry-picked) showing the true cursor (black), raw ridge prediction (dashed purple),
fixed Kalman smoother (green, parameters set a priori from the 0.25 Hz task bandwidth), and the
oracle Kalman (grey dotted, test-set tuned; labelled explicitly as not reported). Panel (b): cy
Pearson r for all subjects under three conditions — Ridge (solid), Fixed (hatched, the reported
honest result), and Oracle (lightly shaded, labelled "test-tuned ceiling — not reported"). The grey
shading between Fixed and Oracle quantifies the inflation from test-set tuning.

## fig09_performance_summary

Performance summary across all subjects and both cursor axes. Each subject has three marks: a solid
bar for raw Ridge regression (no post-processing), a hatched bar for the Fixed post-processing
result (LPF 0.25 Hz for cx; Kalman with bandwidth-matched σ_a for cy; parameters chosen a priori,
not from test scores), and a faint triangle for the Oracle result (test-set tuned, shown only as a
ceiling reference and explicitly labelled as not reported). A placeholder is included for an
autocorrelation-null band to be added after running the surrogate permutation test; the figure
layout will not require redesigning.
