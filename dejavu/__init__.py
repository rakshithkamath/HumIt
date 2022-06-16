import multiprocessing
import os
import sys
import traceback
from itertools import groupby
from time import time
from typing import Dict, List, Tuple

import dejavu.logic.decoder as decoder
from dejavu.base_classes.base_database import get_database
from dejavu.config.settings import (DEFAULT_FS, DEFAULT_OVERLAP_RATIO,
                                    DEFAULT_WINDOW_SIZE, FIELD_FILE_SHA1,
                                    FIELD_TOTAL_HASHES,
                                    FINGERPRINTED_CONFIDENCE,
                                    FINGERPRINTED_HASHES, HASHES_MATCHED,
                                    INPUT_CONFIDENCE, INPUT_HASHES, OFFSET,
                                    OFFSET_SECS, SONG_ID, SONG_NAME, TOPN)
from dejavu.logic.fingerprint import fingerprint

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
import librosa.display
from IPython.display import Audio
import noisereduce as nr

import librosa
import librosa.display
import scipy
import IPython.display as ipd
from scipy.io.wavfile import read,write
from IPython.display import Audio
from scipy.fftpack import fft
import scipy.fftpack as fft
from scipy import signal
import sys, os, wave
from scipy.io import wavfile
from scipy.signal import get_window

def zero_crossing_rate(chunk):
    zero_crossing_rate=librosa.feature.zero_crossing_rate(chunk,frame_length=2048, hop_length=512, center=False)
    return zero_crossing_rate


# # Speectral centroid

# In[117]:


def spectral_centroid(chunk):
    centroid=librosa.feature.spectral_centroid(chunk, sr=22050, S=None, n_fft=2048, hop_length=512, freq=None, win_length=None, window='hann', center=True, pad_mode='reflect')
    return centroid


# # Pitch detection

# In[118]:


def normal_distribution(w):
    width = w+1
    weights = np.exp(-np.square([2*x/width for x in range(width)]))
    weights = np.pad(weights, (width-1,0), 'reflect')
    weights = weights/np.sum(weights)
    return weights

def detect_pitch(int_data,Fs):
    all_pitches=[]
    if 'avg' not in detect_pitch.__dict__:
        detect_pitch.avg = 0
    WIND = 10
    CYCLE = 400
    RATE=Fs
    weights = normal_distribution(WIND)
    windowed_data = np.pad(int_data, WIND, 'reflect')
    smooth_data = np.convolve(int_data, weights, mode='valid')
    smooth_pitches = [0]+[np.mean(smooth_data[:-delay] - smooth_data[delay:]) for delay in range(1,CYCLE)]

    dips = [x for x in range(WIND, CYCLE-WIND) if smooth_pitches[x] == np.min(smooth_pitches[x-WIND:x+WIND])]
    if len(dips) > 1:
        av_dip = np.mean(np.ediff1d(dips))
        cheq_freq = RATE / av_dip
        detect_pitch.avg = detect_pitch.avg*0.5 + cheq_freq*0.5
        all_pitches.append(int(detect_pitch.avg))
        print('\r'+str(int(detect_pitch.avg))+' Hz        ', end='')
    return int(detect_pitch.avg)


# # MFCC calculation

# In[119]:


def mel_frequency(data,Fs):
    FFT_size=2048
    hop_size=15# hop_size in ms
    data = np.pad(data, int(FFT_size / 2), mode='reflect')
    frame_len = np.round(Fs * hop_size / 1000).astype(int)
    frame_num = int((len(data) - FFT_size) / frame_len) + 1
    framed_data = np.zeros((frame_num,FFT_size))

    for n in range(frame_num):
        framed_data[n] = data[n*frame_len:n*frame_len+FFT_size]

    #taking a hanning window and multiplying it to the framed data
    FFT_size=2048
    hanning_window = get_window("hann", FFT_size, fftbins=True)
    windowed_data = framed_data * hanning_window
    ind = 69
    fft_data = np.empty((int(1 + FFT_size // 2), windowed_data.shape[0]), dtype=np.complex64, order='F')

    for n in range(fft_data.shape[1]):
        fft_data[:, n] = fft.fft((windowed_data[n,:]).T, axis=0)[:fft_data.shape[0]]

    fft_data = np.transpose(fft_data)
    data_power=np.square(np.abs(fft_data))
    #get the filter bank splitting points
    freq_min = 0
    freq_max = Fs / 2
    mel_filter_num = 10

    freq_min_mel = 2595.0 * np.log10(1.0 + freq_min / 700.0)
    freq_max_mel = 2595.0 * np.log10(1.0 + freq_max / 700.0)

    mel_freq = np.linspace(freq_min_mel, freq_max_mel, num=mel_filter_num+2)
    freq_set = 700.0 * (10.0**(mel_freq / 2595.0) - 1.0)

    filter_points=np.floor((FFT_size + 1) / Fs * freq_set).astype(int)
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    # taken from the librosa library
    enorm = 2.0 / (mel_freq[2:mel_filter_num+2] - mel_freq[:mel_filter_num])
    filters *= enorm[:, np.newaxis]
    audio_filtered = np.dot(filters, np.transpose(data_power))
    audio_log = 10.0 * np.log10(audio_filtered)
    dct_filter_num = 40
    basis = np.empty((dct_filter_num,mel_filter_num))
    basis[0, :] = 1.0 / np.sqrt(mel_filter_num)

    samples = np.arange(1, 2 * mel_filter_num, 2) * np.pi / (2.0 * mel_filter_num)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / mel_filter_num)

    cepstral_coefficents = np.dot(basis, audio_log)
    l = cepstral_coefficents.shape

    for i in range(0,l[0]):
        for j in range(0,l[1]):
            if np.isnan(cepstral_coefficents[i,j]) or np.isinf(cepstral_coefficents[i,j]):
                cepstral_coefficents[i,j]=0
            
    return cepstral_coefficents

def final_Features(fileName):
    #Loading the song
    data,Fs=librosa.load(fileName, sr=None) #Fs is the sampling frequency of the song
    Audio(data,rate=Fs)
    length = math.ceil(len(data)/(Fs*15))
    chunk=np.array_split(data,length)
    print(Fs)
    print(len(data))
    print(len(chunk))
    print(chunk[0])
    features=[]
    features_zc=[]
    features_sc =[]
    features_p=[]
    features_m=[]
    rdd = nr.reduce_noise(y=chunk[0], sr=Fs)
    fff = mel_frequency(rdd, Fs)
    for i in range(0,len(chunk)):
        #noise reduction
        reduced_noise = nr.reduce_noise(y=chunk[i], sr=Fs)
        #zero_crossing_rate
        zc=zero_crossing_rate(reduced_noise)
        features_zc.append(zc)
        #spectral centroid
        sc=spectral_centroid(reduced_noise)
        features_sc.append(sc)
        #detecting pitch
        pitch=detect_pitch(reduced_noise, Fs)
        features_p.append(pitch)
        #mfcc
        if i != 0:
            mfcc=mel_frequency(reduced_noise,Fs)
            np.resize(mfcc,(40,1000))
            fff+=mfcc
            print(mfcc)
    
    
    
    zc=np.mean(features_zc)
    sc=np.mean(features_sc)
    pitch =np.mean(features_p)

    mfcc=fff
    
    features.append(zc)
    features.append(sc)
    features.append(pitch)
    features.append(mfcc[0:40,0:1000])
    return(features)

def Convert(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return res_dct
    
class Dejavu:
    def __init__(self, config):
        self.config = config

        # initialize db
        db_cls = get_database(config.get("database_type", "mysql").lower())

        self.db = db_cls(**config.get("database", {}))
        self.db.setup()

        # if we should limit seconds fingerprinted,
        # None|-1 means use entire track
        self.limit = self.config.get("fingerprint_limit", None)
        if self.limit == -1:  # for JSON compatibility
            self.limit = None
        self.__load_fingerprinted_audio_hashes()

    def __load_fingerprinted_audio_hashes(self) -> None:
        """
        Keeps a dictionary with the hashes of the fingerprinted songs, in that way is possible to check
        whether or not an audio file was already processed.
        """
        # get songs previously indexed
        self.songs = self.db.get_songs()
        self.songhashes_set = set()  # to know which ones we've computed before
        for song in self.songs:
            song_hash = song[FIELD_FILE_SHA1]
            self.songhashes_set.add(song_hash)

    def get_fingerprinted_songs(self) -> List[Dict[str, any]]:
        """
        To pull all fingerprinted songs from the database.

        :return: a list of fingerprinted audios from the database.
        """
        return self.db.get_songs()

    def delete_songs_by_id(self, song_ids: List[int]) -> None:
        """
        Deletes all audios given their ids.

        :param song_ids: song ids to delete from the database.
        """
        self.db.delete_songs_by_id(song_ids)

    def fingerprint_directory(self, path: str, extensions: str, nprocesses: int = None) -> None:
        """
        Given a directory and a set of extensions it fingerprints all files that match each extension specified.

        :param path: path to the directory.
        :param extensions: list of file extensions to consider.
        :param nprocesses: amount of processes to fingerprint the files within the directory.
        """
        # Try to use the maximum amount of processes if not given.
        try:
            nprocesses = nprocesses or multiprocessing.cpu_count()
        except NotImplementedError:
            nprocesses = 1
        else:
            nprocesses = 1 if nprocesses <= 0 else nprocesses

        pool = multiprocessing.Pool(nprocesses)

        filenames_to_fingerprint = []
        for filename, _ in decoder.find_files(path, extensions):
            # don't refingerprint already fingerprinted files
            if decoder.unique_hash(filename) in self.songhashes_set:
                print(f"{filename} already fingerprinted, continuing...")
                continue
            filenames_to_fingerprint.append(filename)

        # Prepare _fingerprint_worker input
        worker_input = list(zip(filenames_to_fingerprint, [self.limit] * len(filenames_to_fingerprint)))

        # Send off our tasks
        iterator = pool.imap_unordered(Dejavu._fingerprint_worker, worker_input)

        # Loop till we have all of them
        while True:
            try:
                song_name, hashes, file_hash = next(iterator)
            except multiprocessing.TimeoutError:
                continue
            except StopIteration:
                break
            except Exception:
                print("Failed fingerprinting")
                # Print traceback because we can't reraise it here
                traceback.print_exc(file=sys.stdout)
            else:
                sid = self.db.insert_song(song_name, file_hash, len(hashes))
                
                self.db.insert_hashes(sid, hashes)
                self.db.set_song_fingerprinted(sid)
                self.__load_fingerprinted_audio_hashes()

        pool.close()
        pool.join()

    def fingerprint_file(self, file_path: str, song_name: str = None) -> None:
        """
        Given a path to a file the method generates hashes for it and stores them in the database
        for later be queried.

        :param file_path: path to the file.
        :param song_name: song name associated to the audio file.
        """
        song_name_from_path = decoder.get_audio_name_from_path(file_path)
        song_hash = decoder.unique_hash(file_path)
        song_name = song_name or song_name_from_path
        # don't refingerprint already fingerprinted files
        if song_hash in self.songhashes_set:
            print(f"{song_name} already fingerprinted, continuing...")
        else:
            song_name, hashes, file_hash = Dejavu._fingerprint_worker(
                file_path,
                self.limit,
                song_name=song_name
            )
            sid = self.db.insert_song(song_name, file_hash)

            self.db.insert_hashes(sid, hashes)
            self.db.set_song_fingerprinted(sid)
            self.__load_fingerprinted_audio_hashes()

    def generate_fingerprints(self, samples: List[int], Fs=DEFAULT_FS) -> Tuple[List[Tuple[str, int]], float]:
        f"""
        Generate the fingerprints for the given sample data (channel).

        :param samples: list of ints which represents the channel info of the given audio file.
        :param Fs: sampling rate which defaults to {DEFAULT_FS}.
        :return: a list of tuples for hash and its corresponding offset, together with the generation time.
        """
        t = time()
        hashes = fingerprint(samples, Fs=Fs)
        fingerprint_time = time() - t
        return hashes, fingerprint_time

    def find_matches(self, hashes: List[Tuple[str, int]]) -> Tuple[List[Tuple[int, int]], Dict[str, int], float]:
        """
        Finds the corresponding matches on the fingerprinted audios for the given hashes.

        :param hashes: list of tuples for hashes and their corresponding offsets
        :return: a tuple containing the matches found against the db, a dictionary which counts the different
         hashes matched for each song (with the song id as key), and the time that the query took.

        """
        t = time()
        matches, dedup_hashes = self.db.return_matches(hashes)
        query_time = time() - t

        return matches, dedup_hashes, query_time

    def align_matches(self, matches: List[Tuple[int, int]], dedup_hashes: Dict[str, int], queried_hashes: int,
                      topn: int = TOPN) -> List[Dict[str, any]]:
        """
        Finds hash matches that align in time with other matches and finds
        consensus about which hashes are "true" signal from the audio.

        :param matches: matches from the database
        :param dedup_hashes: dictionary containing the hashes matched without duplicates for each song
        (key is the song id).
        :param queried_hashes: amount of hashes sent for matching against the db
        :param topn: number of results being returned back.
        :return: a list of dictionaries (based on topn) with match information.
        """
        # count offset occurrences per song and keep only the maximum ones.
        sorted_matches = sorted(matches, key=lambda m: (m[0], m[1]))
        counts = [(*key, len(list(group))) for key, group in groupby(sorted_matches, key=lambda m: (m[0], m[1]))]
        songs_matches = sorted(
            [max(list(group), key=lambda g: g[2]) for key, group in groupby(counts, key=lambda count: count[0])],
            key=lambda count: count[2], reverse=True
        )

        songs_result = []
        for song_id, offset, _ in songs_matches[0:topn]:  # consider topn elements in the result
            song = self.db.get_song_by_id(song_id)

            song_name = song.get(SONG_NAME, None)
            song_hashes = song.get(FIELD_TOTAL_HASHES, None)
            nseconds = round(float(offset) / DEFAULT_FS * DEFAULT_WINDOW_SIZE * DEFAULT_OVERLAP_RATIO, 5)
            hashes_matched = dedup_hashes[song_id]

            song = {
                SONG_ID: song_id,
                SONG_NAME: song_name.encode("utf8"),
                INPUT_HASHES: queried_hashes,
                FINGERPRINTED_HASHES: song_hashes,
                HASHES_MATCHED: hashes_matched,
                # Percentage regarding hashes matched vs hashes from the input.
                INPUT_CONFIDENCE: round(hashes_matched / queried_hashes, 2),
                # Percentage regarding hashes matched vs hashes fingerprinted in the db.
                FINGERPRINTED_CONFIDENCE: round(hashes_matched / song_hashes, 2),
                OFFSET: offset,
                OFFSET_SECS: nseconds,
                FIELD_FILE_SHA1: song.get(FIELD_FILE_SHA1, None).encode("utf8")
            }

            songs_result.append(song)

        return songs_result

    def recognize(self, recognizer, *options, **kwoptions) -> Dict[str, any]:
        r = recognizer(self)
        return r.recognize(*options, **kwoptions)

    @staticmethod
    def _fingerprint_worker(arguments):
        # Pool.imap sends arguments as tuples so we have to unpack
        # them ourself.
        try:
            file_name, limit = arguments
        except ValueError:
            pass

        song_name, extension = os.path.splitext(os.path.basename(file_name))

        fingerprints, file_hash = Dejavu.get_file_fingerprints(file_name, limit, print_output=True)

        return song_name, fingerprints, file_hash

    @staticmethod
    def get_file_fingerprints(file_name: str, limit: int, print_output: bool = False):
        channels, fs, file_hash = decoder.read(file_name, limit)
        fingerprints = set()
        channel_amount = len(channels)
        for channeln, channel in enumerate(channels, start=1):
            if print_output:
                print(f"Fingerprinting channel {channeln}/{channel_amount} for {file_name}")

            hashes = fingerprint(channel, Fs=fs)

            if print_output:
                print(f"Finished channel {channeln}/{channel_amount} for {file_name}")
            
            fingerprints |= set(hashes)
            #dict = Convert(final_Features(file_name))
            #fingerprints |= set(dict)
        return fingerprints, file_hash
