import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def remove_background(data, background_data):
    if len(data) > len(background_data):
        data = data[:len(background_data)]
    else:
        background_data = background_data[:len(data)]
    return data - background_data

def compute_fourier_transform(filename, background_filename=None, trim_start_sec=0):
    rate, data = wavfile.read(filename)
    
    if background_filename:
        _, background_data = wavfile.read(background_filename)
        data = remove_background(data, background_data)
        
    if trim_start_sec > 0:
        trim_start_samples = int(trim_start_sec * rate)
        data = data[trim_start_samples:]
    
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)
 
    # Remove DC offset
    data = data - np.mean(data)
    
    # Apply Hamming window
    data = data * np.hamming(len(data))
 
    # High-pass filtering to remove frequencies
    data = highpass_filter(data, 4000, rate)
    
    # Compute the Fourier Transform of the audio
    spectrum = np.fft.fft(data)
    
    freq = np.fft.fftfreq(len(spectrum), d=1/rate)
    
    return freq, spectrum

def plot_spectrum(freq, spectrum, title, min_peak_freq=100, error_rate=0.05):
    magnitude = np.abs(spectrum)
    
    pos_indices = np.where(freq > 0)
    pos_freq = freq[pos_indices]
    pos_magnitude = magnitude[pos_indices]
    
    pos_indices_of_interest = np.where(pos_freq >= min_peak_freq)
    pos_freq_of_interest = pos_freq[pos_indices_of_interest]
    pos_magnitude_of_interest = pos_magnitude[pos_indices_of_interest]
    
    peak_index = np.argmax(pos_magnitude_of_interest)
    peak_freq = pos_freq_of_interest[peak_index]
    peak_magnitude = pos_magnitude_of_interest[peak_index]
    
    # Calculate error based on error_rate
    peak_freq_error = error_rate * peak_freq
    peak_magnitude_error = error_rate * peak_magnitude
    
    # Plotting
    plt.plot(freq, magnitude)
    plt.title(title)
    
    # Adding error bar for peak magnitude and frequency
    plt.errorbar(peak_freq, peak_magnitude, xerr=peak_freq_error, yerr=peak_magnitude_error, fmt='o', color='red')
    
    plt.annotate(f'Peak: {peak_freq:.2f} ± {peak_freq_error:.2f} Hz\n'
                 f'Magnitude: {peak_magnitude:.2f} ± {peak_magnitude_error:.2f}',
                 xy=(peak_freq, peak_magnitude),
                 xytext=(peak_freq+1000, peak_magnitude-0.3*peak_magnitude),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(2000, 16000)
    plt.show()
    
    print(f"For {title}: Peak Frequency = {peak_freq:.2f} ± {peak_freq_error:.2f} Hz, "
          f"Peak Magnitude = {peak_magnitude:.2f} ± {peak_magnitude_error:.2f}")

# Paths for your recordings
oak_file = r"C:\Users\deang\OneDrive\Documents\school\3rd year\3008 pro skills\wavfquart\Beech quarter (6).wav"
ash_file = r"C:\Users\deang\OneDrive\Documents\school\3rd year\3008 pro skills\wavfquart\Beech quarter (6).wav"
beech_file = r"C:\Users\deang\OneDrive\Documents\school\3rd year\3008 pro skills\wavfquart\Beech quarter (6).wav"
background_noise_file = r"C:\Users\deang\OneDrive\Documents\school\3rd year\3008 pro skills\wavfquart\Background Final.wav"

trim_seconds = 0  # for example, to trim the first half a second

# Oak Spectrum
plt.figure()  # Create a new figure for Oak Spectrum
freq, oak_spectrum = compute_fourier_transform(oak_file, background_filename=background_noise_file, trim_start_sec=trim_seconds)
plot_spectrum(freq, oak_spectrum, "Oak Spectrum")

# Ash Spectrum
plt.figure()  # Create a new figure for Ash Spectrum
freq, ash_spectrum = compute_fourier_transform(ash_file, background_filename=background_noise_file, trim_start_sec=trim_seconds)
plot_spectrum(freq, ash_spectrum, "Ash Spectrum")

# Beech Spectrum
plt.figure()  # Create a new figure for Beech Spectrum
freq, beech_spectrum = compute_fourier_transform(beech_file, background_filename=background_noise_file, trim_start_sec=trim_seconds)
plot_spectrum(freq, beech_spectrum, "Beech Spectrum")

