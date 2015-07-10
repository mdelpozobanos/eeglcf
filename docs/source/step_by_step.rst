.. _step_by_step:

===================================
EEG artifact rejection step by step
===================================

This is a step-by-step example of how to apply artifact rejection to EEG
data. Note that step 9 is the only phase introduced by LCF, the remaining
steps are defined by any state of the art technique. Note also that LCF
does not modify any of the other steps, hence facilitation its embedding
in any existing method.

1. **Filter the data**: Remove uninteresting frequencies, which can
   potentially carry noise. This usually involves removing some low and high
   frequencies.

2. **Check reference channel**: Make sure that the data has a reference
   channel.

   .. note::
     **Avoid setting a common average reference here.** If there is
     noise within isolated channels, it will spread out to clean sensors with
     common average reference. You can re-reference to whatever reference you
     are interested in after the cleaning process.

3. **Remove the baseline**, i.e. remove the voltage average: Consider the
   rejection of outliers from the computation of the baseline/average, e.g.
   ignoring voltages with a z-score higher than 3.

4. **Reject noisy EEG channels**: Channels that contain mainly noise will
   affect the performance of the BSS algorithm. Keep the list of rejected
   channels, so that you can interpolate them back into the data when the EEG
   is clean.

5. **Reject noisy EEG trials**: Trials that contain mainly noise will affect
   the performance of the BSS algorithm. You will have to compromise here
   between feeding the BSS algorithm with clean data and retaining as much
   trials as possible.

6. **Apply the BSS algorithm**: Project the data from the EEG space to the
   BSS space.

7. **Identify artifactual components**: The literature offers multiple
   approaches to differentiate between artifactual and clean components.
   `ADJUST`_ [Mognon2010]_ and `FASTER`_ [Nolan2010]_ are just two examples.

8. **Process artifactual components**: Process the artifactual components to
   remove the noise. In many cases, artifactual components are entirely rejected
   instead of processed. This is equivalent to set these components to all 0s.

9. **Apply LCF**: Feed the LCF with the original components (computed in step
   6) and the processed components (computed in step 8, if any) corresponding
   to artifactual components (as detected in step 7).

   .. code-block:: python

     c_data = eeglcf.lcf(b_data, a_data)

   Where *b_data* is a numpy.ndarray containing the result of applying a BSS method
   to EEG data, and *a_data* is an alternative "cleaner" version of the previous.
   Both variable have dimensions CxTxE, where C, T and E are the number of
   channels, time samples and events respectively.

10. **Apply the BSS^{-1} algorithm**: Back-project the data from the BSS space
   to the EEG space. This is applied over the set of clean components (not
   detected as artifactual in step 7) and the LCF-processed components (the
   result of step 9).

11. **Remove baseline**. The baseline can be offset by the BSS processing. It
   is recommended to re-remove the baseline to correct it.

12. **Reject channels inside events**: Analyze channels within each event to
   reject and interpolate those that are still too noisy. Note that the
   interpolation is also applied within individual events.

13. **Interpolate channels**: Interpolate channels rejected in step 4 back
   into the data.

Once the EEG is clean, you may apply any further processing methods, such as
changing the reference to common average.

The method :func:'function_name' *eeg_art_rej* is an example implementation of the above process.

References
----------
[Mognon2010] A. Mognon, Jovicich J., Bruzzone L., and Buiatti M. Adjust: An
    automatic eeg artifact detector based on the joint use of spatial and
    temporal features. Psychophysiology, pages 229-240, July 2010.
[Nolan2010] H. Nolan, R. Whelan, and R.B. Reilly. Faster: Fully automated
    statistical thresholding for eeg artifact rejection. Journal of
    Neuroscience Methods, 192(1):152-162, 2010.
.. _ADJUST: https://github.com/mdelpozobanos/eegadjust
.. _FASTER: https://github.com/mdelpozobanos/eegfaster
