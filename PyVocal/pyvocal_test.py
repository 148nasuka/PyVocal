import librosa  # 音楽解析ライブラリ(要、管理者権限)
import librosa.display  # 解析出力用
import pyworld as pw  # 高精度フーリエ変換ライブラリ
from pydub import AudioSegment  # 音声加工ライブラリ（接合）
import soundfile as sf  # 音声ライブラリ（読み込み、書き出し）
from matplotlib import pyplot as plt  # 数学系ライブラリ（プロット）
import numpy as np  # 行列演算系ライブラリ
import argparse  # 引数解析系ライブラリ
import xml.etree.ElementTree as ET  # MusicXML解析用
import csv  # csv解析用


# 簡易歌唱合成音声（波形接続型）
class flat_synthesize:

    def __init__(self):
        # self.filename = './input/' + input().strip()  + '.musicxml'
        self.filename = './input/track1.musicxml'  # MusicXML楽譜データ
        self.BPM = 0  # テンポ
        self.quarter_note = 0  # 基準四分音符 = ms
        self.lyrics = []  # 歌詞データ
        self.notes = []  # 音程データ
        self.notes_octaves = []  # 音程データ（オクターブ）
        self.notes_key = ''  # スケール判定
        self.notes_alter = []  # フラットかシャープか
        self.notes_dot = []  # 付点の有無
        self.actual_notes = []  # 連符の種類
        self.staccato_notes = []
        self.notes_rest = []  # 休符の種類
        self.note_conditions = []  # 音符のアタックかリリースか
        self.note_type = []  # 音符の種類
        self.note_freq = []  # 音符の周波数
        self.time_line = np.array([self.notes, self.note_type, self.note_conditions,
                                   self.note_freq, self.lyrics, self.notes_dot,
                                   self.notes_rest, self.actual_notes])
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-f", "--frame_period", type=float, default=5.0)
        self.parser.add_argument("-s", "--speed", type=int, default=1)
        self.args = self.parser.parse_args()

    # MusicXMLから音声加工に必要な情報をとってくる
    def read_XML(self):
        tree = ET.parse(self.filename)  # 入力楽譜ファイル
        root = tree.getroot()
        measure_count = 0

        i = 0
        for name in root.iter('sound'):  # テンポ読み込み
            if i == 0:
                self.BPM = name.attrib['tempo']
                i = 1

        for name in root.iter('fifths'):  # スケール読み込み
            self.notes_key = name.text

        for _ in root.iter('measure'):  # 小節数カウント
            measure_count = measure_count + 1

        for i in range(0, measure_count, 1):
            measure = root[5][i]  # 小節区切り

            for child in measure:
                if child.find('type') is not None:
                    self.note_type.append(child.find('type').text)
                else:
                    self.note_type.append('Null')

                if child.find('lyric') is not None:
                    self.lyrics.append(child.find('lyric').find('text').text)
                else:
                    self.lyrics.append('Null')

                if child.find('tie') is not None:
                    self.note_conditions.append(child.find('tie').attrib['type'])
                else:
                    self.note_conditions.append('Null')

                if child.find('pitch') is not None:
                    self.notes_octaves.append(child.find('pitch').find('octave').text)
                else:
                    self.notes_octaves.append('Null')

                if child.find('pitch') is not None:
                    self.notes.append(child.find('pitch').find('step').text)
                else:
                    self.notes.append('Null')

                if child.find('dot') is not None:
                    self.notes_dot.append('dot')
                else:
                    self.notes_dot.append('Null')

                if child.find('time-modification') is not None:
                    self.actual_notes.append(child.find('time-modification').find('actual-notes').text)
                else:
                    self.actual_notes.append('Null')

                if child.find('rest') is not None:
                    self.notes_rest.append('rest')
                else:
                    self.notes_rest.append('Null')

                if child.find('staccato') is not None:
                    self.staccato_notes.append('staccato')
                else:
                    self.staccato_notes.append('Null')

        for i in range(0, len(self.notes), 1):  # 上記の音程を統合
            if self.notes_key == '0':
                self.notes[i] = self.notes_octaves[i] + self.notes[i]
            elif self.notes_key == '1':
                if self.notes[i] == 'F':
                    self.notes[i] = self.notes_octaves[i] + self.notes[i] + 's'
                else:
                    self.notes[i] = self.notes_octaves[i] + self.notes[i]
            elif self.notes_key == '2':
                if self.notes[i] == 'F' or self.notes[i] == 'C':
                    self.notes[i] = self.notes[i] + self.notes_octaves[i] + 's'
                else:
                    self.notes[i] = self.notes_octaves[i] + self.notes[i]
            elif self.notes_key == '3':
                if self.notes[i] == 'F' or self.notes[i] == 'C' or self.notes[i] == 'G':
                    self.notes[i] = self.notes_octaves[i] + self.notes[i] + 's'
                else:
                    self.notes[i] = self.notes_octaves[i] + self.notes[i]
            elif self.notes_key == '4':
                if self.notes[i] == 'F' or self.notes[i] == 'C' or self.notes[i] == 'G' or self.notes[i] == 'D':
                    self.notes[i] = self.notes_octaves[i] + self.notes[i] + 's'
                else:
                    self.notes[i] = self.notes_octaves[i] + self.notes[i]
            elif self.notes_key == '5':
                if self.notes[i] == 'F' or self.notes[i] == 'C' or self.notes[i] == 'G' or self.notes[i] == 'D' \
                        or self.notes[i] == 'A':
                    self.notes[i] = self.notes_octaves[i] + self.notes[i] + 's'
                else:
                    self.notes[i] = self.notes_octaves[i] + self.notes[i]
            elif self.notes_key == '6':
                if self.notes[i] == 'F' or self.notes[i] == 'C' or self.notes[i] == 'G' or self.notes[i] == 'D' \
                        or self.notes[i] == 'A' or self.notes[i] == 'E':
                    self.notes[i] = self.notes_octaves[i] + self.notes[i] + 's'
                else:
                    self.notes[i] = self.notes_octaves[i] + self.notes[i]
            elif self.notes_key == '-1':
                if self.notes[i] == 'B':
                    self.notes[i] = self.notes_octaves[i] + self.notes[i] + 'f'
                else:
                    self.notes[i] = self.notes_octaves[i] + self.notes[i]
            elif self.notes_key == '-2':
                if self.notes[i] == 'B' or self.notes[i] == 'E':
                    self.notes[i] = self.notes_octaves[i] + self.notes[i] + 'f'
                else:
                    self.notes[i] = self.notes_octaves[i] + self.notes[i]
            elif self.notes_key == '-3':
                if self.notes[i] == 'B' or self.notes[i] == 'E' or self.notes[i] == 'A':
                    self.notes[i] = self.notes_octaves[i] + self.notes[i] + 'f'
                else:
                    self.notes[i] = self.notes_octaves[i] + self.notes[i]
            elif self.notes_key == '-4':
                if self.notes[i] == 'B' or self.notes[i] == 'E' or self.notes[i] == 'A' or self.notes[i] == 'D':
                    self.notes[i] = self.notes_octaves[i] + self.notes[i] + 'f'
                else:
                    self.notes[i] = self.notes_octaves[i] + self.notes[i]
            elif self.notes_key == '-5':
                if self.notes[i] == 'B' or self.notes[i] == 'E' or self.notes[i] == 'A' or self.notes[i] == 'D' \
                        or self.notes[i] == 'G':
                    self.notes[i] = self.notes_octaves[i] + self.notes[i] + 'f'
                else:
                    self.notes[i] = self.notes_octaves[i] + self.notes[i]
            elif self.notes_key == '-6':
                if self.notes[i] == 'B' or self.notes[i] == 'E' or self.notes[i] == 'A' or self.notes[i] == 'D' \
                        or self.notes[i] == 'G' or self.notes[i] == 'C':
                    self.notes[i] = self.notes_octaves[i] + self.notes[i] + 'f'
                else:
                    self.notes[i] = self.notes_octaves[i] + self.notes[i]

        for i in range(len(self.notes)):
            freq_list = open("./library/freq_list.csv", "r", encoding="utf-8", errors="", newline="")
            reader = csv.reader(freq_list)
            if self.notes[i] != 'NullNull':
                freq = [line for line in reader if self.notes[i] in line]
                self.note_freq.append(freq[0][1])
            else:
                self.note_freq.append(0)

        # 時間別で処理できるように配列更新
        self.time_line = np.array([self.notes, self.note_type, self.note_conditions,
                                   self.note_freq, self.lyrics, self.notes_dot,
                                   self.notes_rest, self.actual_notes, self.staccato_notes])

        self.quarter_note = 1000 / (int(self.BPM) / 60)

        # デバッグプリント
        print('BPM : ' + self.BPM)
        print('Lyrics : ' + ','.join(self.lyrics))
        print('Note_conditions : ' + ', '.join(self.note_conditions))
        print('Note_type : ' + ', '.join(self.note_type))
        print('Notes : ' + ', '.join(self.notes))
        print('Dots : ' + ', '.join(self.notes_dot))
        print('Details of the 0th note : ' + ', '.join(self.time_line[:, 7]))
        print(len(self.time_line))
        print(len(self.notes))

    # 楽譜から合成音声を生成(world)
    def worldsynth(self):

        out = AudioSegment.from_file("./library/normal/empty.wav", format="wav")
        out = out[:100]

        for i in range(len(self.notes)):
            note = self.time_line[0][i]
            now_lyrics = self.time_line[4][i]
            select_path = ""

            freq_list = open("./library/freq_list.csv", "r", encoding="utf-8", errors="", newline="")
            reader = csv.reader(freq_list)

            if now_lyrics != "Null" and str(note) != "NullNull":
                tone = [line for line in reader if str(note) in line]
                select_tone = tone[0][2]
                select = select_tone + "_" + now_lyrics + ".wav"
                select_path = "./library/normal/" + select_tone + "_" + now_lyrics + ".wav"
            else:
                select = "empty.wav"
                select_path = "./library/normal/empty.wav"

            x, fs = sf.read(select_path)

            print(select)

            # 合成
            f0, sp, ap = pw.wav2world(x, fs)

            # 推定パラメータ設定 周波数窓、時間窓
            _f0, t = pw.dio(x, fs, f0_floor=100.0, f0_ceil=10000.0,
                            channels_in_octave=5,
                            frame_period=self.args.frame_period,
                            speed=self.args.speed,
                            allowed_range=20)
            _sp = pw.cheaptrick(x, _f0, t, fs)
            _ap = pw.d4c(x, _f0, t, fs)

            lest_sp = np.ones_like(sp)
            robot_like_f0 = np.zeros_like(f0)

            num = self.time_line[3][i]

            # F0ノーマライズ
            for f in range(1000):
                plus = float(num) - _f0[f]
                robot_like_f0[f] = _f0[f] + plus

            # 部分書き出し
            _y = pw.synthesize(robot_like_f0, _sp, _ap * 0.0, fs, self.args.frame_period)

            sf.write('./output/preout/' + str(i) + ".wav", _y, fs)

            start_time = open("./library/normal/start_time.csv", "r", encoding="utf-8", errors="", newline="")
            reader = csv.reader(start_time)
            start = [line for line in reader if select in line]
            start_ms = int(start[0][1])

            end_ms = 0

            if i < len(self.notes) - 1:
                end_ms = int(start[0][9])
                if self.time_line[2][i] == "stop":
                    end_ms = int(start[0][3])

            load = AudioSegment.from_file("./output/preout/" + str(i) + ".wav", format="wav")
            load_prev = AudioSegment.from_file("./output/preout/" + str(i) + ".wav", format="wav")
            if i > 1:
                print(str(i) + " / " + str(len(self.notes)))
                load_prev = AudioSegment.from_file("./output/preout/" + str(i - 1) + ".wav", format="wav")

            print(self.time_line[0][i])

            # 休符ではないかつスラ―ではないとき
            if self.time_line[0][i] != "NullNull" and self.time_line[2][i] == "Null":
                # 通常ノート
                if self.time_line[1][i] == "32nd" and self.time_line[5][i] == "Null" \
                        and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 10 + self.quarter_note / 16], crossfade=10)
                    out = out.append(load[end_ms - 100 - self.quarter_note / 16: end_ms])
                elif self.time_line[1][i] == "16th" and self.time_line[5][i] == "Null" \
                        and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 10 + self.quarter_note / 8], crossfade=10)
                    out = out.append(load[end_ms - 100 - self.quarter_note / 8: end_ms])
                elif self.time_line[1][i] == "eighth" and self.time_line[5][i] == "Null" \
                        and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 10 + self.quarter_note / 4], crossfade=10)
                    out = out.append(load[end_ms - 100 - self.quarter_note / 4: end_ms])
                elif self.time_line[1][i] == "quarter" and self.time_line[5][i] == "Null" \
                        and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 10 + self.quarter_note / 2], crossfade=10)
                    out = out.append(load[end_ms - 100 - self.quarter_note / 2: end_ms])
                elif self.time_line[1][i] == "half" and self.time_line[5][i] == "Null" \
                        and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 10 + self.quarter_note], crossfade=10)
                    out = out.append(load[end_ms - 100 - self.quarter_note: end_ms])
                elif self.time_line[1][i] == "whole" and self.time_line[5][i] == "Null" \
                        and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 10 + self.quarter_note * 2], crossfade=10)
                    out = out.append(load[end_ms - 100 - self.quarter_note * 2: end_ms])
                # 符点ノート
                elif self.time_line[1][i] == "32nd" and self.time_line[5][i] == "dot" \
                        and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 100 + self.quarter_note / 8 + self.quarter_note / 16])
                    out = out.append(load[end_ms - self.quarter_note / 4 - self.quarter_note / 8: end_ms],
                                     crossfade=self.quarter_note / 8 + self.quarter_note / 16)
                elif self.time_line[1][i] == "16th" and self.time_line[5][i] == "dot" \
                        and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 100 + self.quarter_note / 4 + self.quarter_note / 8])
                    out = out.append(load[end_ms - self.quarter_note / 2 - self.quarter_note / 4: end_ms],
                                     crossfade=self.quarter_note / 4 + self.quarter_note / 8)
                elif self.time_line[1][i] == "eighth" and self.time_line[5][i] == "dot" \
                        and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 100 + (self.quarter_note * 3) / 8])
                    out = out.append(load[end_ms - (self.quarter_note * 3) / 4: end_ms],
                                     crossfade=(self.quarter_note * 3) / 8)
                elif self.time_line[1][i] == "quarter" and self.time_line[5][i] == "dot" \
                        and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 100 + self.quarter_note + self.quarter_note / 2])
                    out = out.append(load[end_ms - self.quarter_note * 2 - self.quarter_note: end_ms],
                                     crossfade=self.quarter_note + self.quarter_note / 2)
                elif self.time_line[1][i] == "half" and self.time_line[5][i] == "dot" \
                        and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 100 + self.quarter_note * 2 + self.quarter_note])
                    out = out.append(load[end_ms - self.quarter_note * 4 - self.quarter_note * 2: end_ms],
                                     crossfade=self.quarter_note * 2 + self.quarter_note)
                elif self.time_line[1][i] == "whole" and self.time_line[5][i] == "dot" \
                        and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + self.quarter_note * 4 + self.quarter_note * 2])
                    out = out.append(load[end_ms - self.quarter_note * 8 - self.quarter_note * 4: end_ms],
                                     crossfade=self.quarter_note * 4 + self.quarter_note * 2)
                # 連符ノート
                elif self.time_line[1][i] == "32nd" and self.time_line[5][i] == "Null" \
                        and self.time_line[7][i] == "3":
                    out = out.append(load[start_ms: start_ms + 100 + (self.quarter_note / 4 / 3) / 2])
                    out = out.append(load[end_ms - 100 - (self.quarter_note / 4 / 3) / 2: end_ms])
                elif self.time_line[1][i] == "16th" and self.time_line[5][i] == "Null" \
                        and self.time_line[7][i] == "3":
                    out = out.append(load[start_ms: start_ms + 100 + (self.quarter_note / 2 / 3) / 2])
                    out = out.append(load[end_ms - 100 - (self.quarter_note / 2 / 3) / 2: end_ms])
                elif self.time_line[1][i] == "eighth" and self.time_line[5][i] == "Null" \
                        and self.time_line[7][i] == "3":
                    out = out.append(load[start_ms: start_ms + 100 + (self.quarter_note / 3) / 2])
                    out = out.append(load[end_ms - 100 - (self.quarter_note / 3) / 2: end_ms])
                elif self.time_line[1][i] == "quarter" and self.time_line[5][i] == "Null" \
                        and self.time_line[7][i] == "3":
                    out = out.append(load[start_ms: start_ms + 100 + ((self.quarter_note * 2) / 3) / 2])
                    out = out.append(load[end_ms - 100 - ((self.quarter_note * 2) / 3) / 2: end_ms])
            # スラー開始時
            if self.time_line[2][i] == "start":
                if self.time_line[1][i] == "32nd" and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 100 + self.quarter_note / 8])
                elif self.time_line[1][i] == "16th" and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 100 + self.quarter_note / 4])
                elif self.time_line[1][i] == "eighth" and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 100 + self.quarter_note / 2])
                elif self.time_line[1][i] == "quarter" and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 100 + self.quarter_note])
                elif self.time_line[1][i] == "half" and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 100 + self.quarter_note * 2])
                elif self.time_line[1][i] == "whole" and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 100 + self.quarter_note * 4])
                # 連符ノート
                elif self.time_line[1][i] == "32nd" and self.time_line[7][i] == "3":
                    out = out + load[start_ms: start_ms + 100 + (self.quarter_note / 4 / 3)]
                elif self.time_line[1][i] == "16th" and self.time_line[7][i] == "3":
                    out = out.append(load[start_ms: start_ms + 100 + (self.quarter_note / 2 / 3)])
                elif self.time_line[1][i] == "eighth" and self.time_line[7][i] == "3":
                    out = out.append(load[start_ms: start_ms + 100 + (self.quarter_note / 3)])
                elif self.time_line[1][i] == "quarter" and self.time_line[7][i] == "3":
                    out = out.append(load[start_ms: start_ms + 100 + ((self.quarter_note * 2) / 3)])
            # スラー終了時
            if self.time_line[2][i] == "stop":
                if self.time_line[1][i] == "32nd" and self.time_line[7][i] == "Null":
                    out = out.append(load[end_ms - self.quarter_note / 4: end_ms],
                                     crossfade=self.quarter_note / 8).fade_out(100)
                elif self.time_line[1][i] == "16th" and self.time_line[7][i] == "Null":
                    out = out.append(load[end_ms - self.quarter_note / 2: end_ms],
                                     crossfade=self.quarter_note / 4).fade_out(100)
                elif self.time_line[1][i] == "eighth" and self.time_line[7][i] == "Null":
                    out = out.append(load[end_ms - self.quarter_note: end_ms],
                                     crossfade=self.quarter_note / 2).fade_out(100)
                elif self.time_line[1][i] == "quarter" and self.time_line[7][i] == "Null":
                    for j in range(2):
                        out = out.append(load[end_ms - self.quarter_note: end_ms],
                                         crossfade=self.quarter_note / 2).fade_out(100)
                elif self.time_line[1][i] == "half" and self.time_line[7][i] == "Null":
                    for j in range(4):
                        out = out.append(load[end_ms - self.quarter_note: end_ms],
                                         crossfade=self.quarter_note / 2).fade_out(100)
                elif self.time_line[1][i] == "whole" and self.time_line[7][i] == "Null":
                    for j in range(8):
                        out = out.append(load[end_ms - self.quarter_note: end_ms],
                                         crossfade=self.quarter_note / 2).fade_out(100)
                # 連符ノート
                elif self.time_line[1][i] == "32nd" and self.time_line[7][i] == "3":
                    out = out.append(load[end_ms - 100 - (self.quarter_note / 4 / 3): end_ms])
                elif self.time_line[1][i] == "16th" and self.time_line[7][i] == "3":
                    out = out.append(load[end_ms - 100 - (self.quarter_note / 2 / 3): end_ms])
                elif self.time_line[1][i] == "eighth" and self.time_line[7][i] == "3":
                    out = out.append(load[end_ms - 100 - (self.quarter_note / 3): end_ms])
                elif self.time_line[1][i] == "quarter" and self.time_line[7][i] == "3":
                    out = out.append(load[end_ms - 100 - ((self.quarter_note * 2) / 3): end_ms])
            # 休符の時
            if self.time_line[6][i] == "rest" and self.time_line[2][i - 1] != "start":
                if self.time_line[1][i] == "32nd" and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 100 + self.quarter_note / 8])
                elif self.time_line[1][i] == "16th" and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 100 + self.quarter_note / 4])
                elif self.time_line[1][i] == "eighth" and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 100 + self.quarter_note / 2])
                elif self.time_line[1][i] == "quarter" and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 100 + self.quarter_note])
                elif self.time_line[1][i] == "half" and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 100 + self.quarter_note * 2])
                elif (self.time_line[1][i] == "whole" or self.time_line[1][i] == "Null") \
                        and self.time_line[7][i] == "Null":
                    out = out.append(load[start_ms: start_ms + 100 + self.quarter_note * 4])
                elif self.time_line[1][i] == "32nd" and self.time_line[7][i] == "3":
                    out = out.append(load[start_ms: start_ms + 100 + (self.quarter_note / 4) / 3])
                elif self.time_line[1][i] == "16th" and self.time_line[7][i] == "3":
                    out = out.append(load[start_ms: start_ms + 100 + (self.quarter_note / 2) / 3])
                elif self.time_line[1][i] == "eighth" and self.time_line[7][i] == "3":
                    out = out.append(load[start_ms: start_ms + 100 + self.quarter_note / 3])
                elif self.time_line[1][i] == "quarter" and self.time_line[7][i] == "3":
                    out = out.append(load[start_ms: start_ms + 100 + (self.quarter_note * 2) / 3])
            elif self.time_line[6][i] == "rest" and self.time_line[2][i - 1] == "start":
                if self.time_line[1][i] == "32nd" and self.time_line[7][i] == "3":
                    out = out.append(load_prev[end_ms - 100 - self.quarter_note / 4: end_ms])
                elif self.time_line[1][i] == "16th" and self.time_line[7][i] == "3":
                    out = out.append(load_prev[end_ms - 100 - self.quarter_note / 2: end_ms])
                elif self.time_line[1][i] == "eighth" and self.time_line[7][i] == "3":
                    out = out.append(load_prev[start_ms + 100: start_ms + 200 + (self.quarter_note / 3)])
                elif self.time_line[1][i] == "quarter" and self.time_line[7][i] == "3":
                    out = out.append(load_prev[start_ms + 100: start_ms + 200 + ((self.quarter_note * 2) / 3)])
        out.export("./output/preout.wav", format="wav")

    # 生成した合成音声のピッチ補間
    def legato_audio(self):
        x, fs = sf.read('./output/preout.wav')

        f0, t = pw.dio(x, fs, f0_floor=70.0, f0_ceil=10000.0,
                       channels_in_octave=5,
                       frame_period=self.args.frame_period,
                       speed=self.args.speed,
                       allowed_range=10)
        sp = pw.cheaptrick(x, f0, t, fs)
        ap = pw.d4c(x, f0, t, fs)

        ip_f0 = f0

        print(f0.size)
        print(f0.ndim)
        t = 1
        count = 0

        for f in range(0, ip_f0.size, 1):
            if ip_f0[f - 10] > 150:
                select = ip_f0[f - 10] - ip_f0[f]
                target_pit = ip_f0[f]
                if select > 20:
                    count = count + 1
                if count > 5:
                    recent_pit = ip_f0[f - 10]
                    print("downpitch_detect")
                    print(target_pit)
                    ip_f0[f - 5] = recent_pit - t
                    t = t + 1
                    if ip_f0[f - 5] < target_pit:
                        ip_f0[f - 5] = target_pit
                        count = 0
                        t = 1

        t = 1
        count = 0

        for f in range(0, ip_f0.size, 1):
            if ip_f0[f - 10] > 150:
                select = ip_f0[f - 10] - ip_f0[f]
                target_pit = ip_f0[f]
                if select < -20:
                    count = count - 1
                if count < -5:
                    recent_pit = ip_f0[f - 10]
                    print("uppitch_detect")
                    print(target_pit)
                    ip_f0[f - 5] = recent_pit + t
                    t = t + 1
                    if ip_f0[f - 5] > target_pit:
                        ip_f0[f - 5] = target_pit
                        count = 0
                        t = 1

        # 最終書き出し
        y = pw.synthesize(ip_f0, sp, ap * 0.0, fs, self.args.frame_period)
        sf.write('./output/out.wav', y, fs)

        # 解析表示(liblosa)
        path = "./output/preout.wav"
        y1, sr1 = librosa.load(path)
        S1 = librosa.feature.melspectrogram(y1, sr=sr1, n_mels=128)
        log_S1 = librosa.power_to_db(S1, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_S1, sr=sr1, x_axis='time', y_axis='mel')
        plt.title('repitched voice')
        plt.colorbar(format='%+02.0f dB')
        plt.show()


FS = flat_synthesize()
FS.read_XML()
FS.worldsynth()
FS.legato_audio()
