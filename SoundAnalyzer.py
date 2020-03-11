import cmath
import pyaudio
import numpy as np
import matplotlib.pyplot as plt

class SoundAnalyzer:
    '''
    音を解析するクラス
    '''
    def __init__(self, rate, channel=1, chunk=2048):
        self.rate = rate
        self.chunk = chunk
        self.sleep_time = 0.01
        # 音取り込み準備
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=channel, rate=rate, input=True, frames_per_buffer=chunk)
        # 周波数領域で必要な値を準備
        self.freq_length = int(self.chunk / 2)
        self.freq_list = [i * self.rate / self.chunk for i in range(self.freq_length)]
        # グラフ描画準備
        _, (self.ax_l, self.ax_r) = plt.subplots(ncols=2, figsize=(10, 4))
    
    def dft(self, f):
        '''
        離散フーリエ変換
        '''
        n = len(f)
        A = np.arange(n)
        M = cmath.e**(-1j * A.reshape(1, -1) * A.reshape(-1, 1) * 2 * cmath.pi / n)
        return np.sum(f * M, axis=1)

    def idft(self, f):
        '''
        離散逆フーリエ変換
        '''
        n = len(f)
        A = np.arange(n)
        M = cmath.e**(1j * A.reshape(1, -1) * A.reshape(-1, 1) * 2 * cmath.pi / n)
        return np.sum(f * M, axis=1) / n

    def run(self):
        try:
            print('*** start ***\nPlease input ctrl + c, if you would like to end : ', end='')
            while True:
                # 音の取り込み
                data = self.stream.read(self.chunk, exception_on_overflow=False)

                # 時間領域での音データ
                sound_data = np.frombuffer(data, dtype='int16')
                self.ax_l.plot(range(self.chunk), sound_data)
                self.ax_l.set_xlabel('Time')
                self.ax_l.set_ylim([-20000, 20000])

                # 周波数領域での音データ
                fft_data = self.dft(sound_data)
                amp_data = np.abs(fft_data/self.freq_length)
                # ナイキストの定理より、freq_lengthより大きい範囲は不要
                # 0はDC成分なので非表示とする
                self.ax_r.plot(self.freq_list[1:self.freq_length], amp_data[1:self.freq_length])
                self.ax_r.set_xlabel('Frequency[Hz]')
                self.ax_r.set_ylim(-5, 5000)

                # matplotlibのリアルタイム描画
                plt.draw()
                plt.pause(self.sleep_time)
                # グラフの初期化
                self.ax_l.cla()
                self.ax_r.cla()
        except KeyboardInterrupt:
            print('\n*** end ***')
            # お片付け
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            plt.close()

if __name__ == "__main__":
    sa = SoundAnalyzer(44100)
    sa.run()