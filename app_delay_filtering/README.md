#### The Variation Threshold Depends On

1. Delay Distribution (Main Factor)

- The delay distribution $g(t)$ acts like a low-pass filter.
- Its Fourier transform $\hat g(\omega)$ tells us which frequencies (oscillations in β or infections) pass through.
- It defines a cutoff frequency $\omega_c$ beyond which β(t) fluctuations are unidentifiable — they get smoothed away.

This is independent of the specific confirmed case curve. Only β(t) components with frequencies below this window are theoretically recoverable.

2. Confirmed Cases Curve (Affects Realistic Detectability)
While the delay distribution defines the theoretical limit, the actual confirmable information also depends on:

- The shape and energy of the infection signal (which comes from β(t)),
- The magnitude of the signal (e.g., low vs high case counts),
- The noise level in the data (real-world observation error).

So, even if a frequency is “passable” by $\hat g(\omega)$, if the corresponding component of β(t) is too weak (low amplitude), it may not be distinguishable from noise.

From the convolution relationship in the frequency domain:
$$
\hat{C}(\omega) = \hat{I}(\omega) \cdot \hat{g}(\omega)
$$

Now we want to explore: for which frequencies does $ \hat{g}(\omega) \approx 0 $, such that changes in $ \hat{I}(\omega) $ (or equivalently $ \beta(t) $ ) become invisible in the confirmed cases?


By plotting the magnitude response of the delay distribution in the frequency domain $
|\hat{g}(\omega)|,
$

we obtain the attenuation profile of the delay kernel. This profile indicates how different frequency components of the infection signal are preserved or suppressed due to the delay structure.

To determine a cutoff frequency, we define a threshold (e.g., 10\% ? ):
$$
|\hat{g}(\omega)| < 0.1
$$
Frequencies satisfying this condition are considered \emph{unidentifiable} because any corresponding variations in $ \beta(t) $ will be strongly attenuated in the observed confirmed case data.

The cutoff frequency $\omega_c $ where $ |\hat{g}(\omega)| $ drops below the threshold defines the theoretical limit of identifiability under the given delay distribution.

| Method                        | Theoretical or Empirical | Depends on delay? | Depends on cases? | Other notes                           |
| ----------------------------- | ------------------------ | ----------------- | ----------------- | ----------------------------------- |
| Fourier cutoff from $\hat{g}$ | Theoretical              | yes             | no              | General limit                       |
| Noise sweep experiment        | Empirical                | yes             | yes             | Realistic identifiability threshold |
| Sinusoidal injection test     | Semi-empirical           | yes             | yes             | Frequency-specific detectability    |
