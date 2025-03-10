import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from scipy.signal import lfilter, freqz

class formant():

    def __init__(self, audio_file_name, bit_rate = None, start_time = 0, duration = None, BWcutoff = 1000):
        self.s, self.sr = librosa.load(audio_file_name,
                                       sr = bit_rate,
                                       offset =  start_time,
                                       duration = duration,
                                       dtype = np.float64)

        self.f0, self.voiced_flag, self.voiced_probs = librosa.pyin(self.s,
                                                                    sr = self.sr,
                                                                    fmin = librosa.note_to_hz('C2'),
                                                                    fmax = librosa.note_to_hz('C7'))
        self.M, self.F = self.get_F(self.s, self.sr, BWcutoff)

        display(self.table(self.f0, self.F))
        self.plot_F(self.s, self.sr, np.insert(self.F, 0, 0), self.M)
        

    def get_F(self, s, sr, BWcutoff):
        M = int(sr/1000+4)
        a = librosa.lpc(s, order = M)

        F = np.sort(np.angle(np.roots(a)))
        order = np.argsort(np.angle(np.roots(a)))
        F = F*sr/(2*np.pi);

        BW = -2*np.log(abs(np.roots(a)))*sr/(2*np.pi);
        BW = BW[order];

        idx = [i for i in range(F.shape[0]) if (0 < F[i] < (sr/2)) & (BW[i] < BWcutoff)]

        return M, F[idx]

    def table(self, f0, F):
        data = [int(f) for f in np.insert(F, 0, np.nanmean(f0)) if f<5000]

        df = pd.DataFrame(data, columns = ['진동수 (Hz)'])
        df = df.transpose()
        df = df.add_prefix('F')

        return df


    def plot_F(self, s, sr, F = [0], M = 20, xlim = 5000, ylim = 5000):
        a = librosa.lpc(s, order = M)
        w, h = freqz(1, a)

        D = librosa.stft(s)
        S_db = librosa.amplitude_to_db(np.abs(D), ref = np.max)

        x, y = w*sr/(2*np.pi), 20*np.log10(np.abs(h))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
        ax1.plot(x, y)
        ax1.set_xlim(0, xlim)
        ax1.set_ylim(np.min(y[np.where(x>5000)[0][0]-1]),)
        ax1.set_yticks([])
        ax1.set_xlabel('Hz')

        if len(F)>1:
            for f in F:
                if (0 < f < xlim):
                    idx = np.where(x>f)[0][0]
                    ymax = np.max([y[idx-1],y[idx],y[idx+1]])
                    ax1.text(f,ymax,str(int(f))+'Hz')
                else:
                    pass
        
        img = librosa.display.specshow(S_db, sr = sr, x_axis='time', y_axis='linear', ax=ax2)
        ax2.set_ylim(0, ylim)
        ax2.set_yticks(np.linspace(0, ylim, 11))
        
        plt.show()
