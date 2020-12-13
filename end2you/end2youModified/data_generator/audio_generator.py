import numpy as np
import h5py

from pathlib import Path
from moviepy.editor import AudioFileClip
from .generator import Generator
from .file_reader import FileReader


class AudioGenerator(Generator):
    
    def __init__(self, 
                 labelfile_reader:FileReader, 
                 fps:int = 16000, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fps = fps
        self.labelfile_reader = labelfile_reader
    
    def _get_samples(self, data_file:str, label_file:str):

        file_data, attrs_name = self.labelfile_reader.read_file(label_file)
        file_data = np.array(file_data).astype(np.float32)

        clip = AudioFileClip(str(data_file), fps=self.fps)

        # CHANGED (sample duration of 40 ms (or 100 ms))
        # Every sample has the same label
        #sample_duration = 0.04
        sample_duration = 1
        num_samples = int(self.fps * sample_duration)

        seq_num = int(clip.duration/sample_duration)

        labels = np.repeat(file_data, seq_num, axis=0)

        frames = []        
        for i in range(seq_num):
            start_time = i * sample_duration
            end_time = (i+1) * sample_duration
            
            data_frame = np.array(list(clip.subclip(start_time, end_time).iter_frames()))
            data_frame = data_frame.mean(1)[:num_samples]
            
            frames.append(data_frame.astype(np.float32))
        
        frames = np.array(frames).astype(np.float32)
        labels = np.array(labels).astype(np.float32)

        #print(frames.shape)
        #print(frames[0])
        #print(labels)
        #print(labels.shape)
        #print(data_file)
        #print(seq_num)
        #print(num_samples)
        #print(attrs_name)

        return frames, labels, seq_num, num_samples, attrs_name
    
    def serialize_samples(self, writer:h5py.File, data_file:str, label_file:str):
        frames, labels, seq_num, num_samples, names = self._get_samples(data_file, label_file)
        
        # store data
        writer.create_dataset('audio', data=frames)
        writer.create_dataset('labels', data=labels)
        
        # Save meta-data
        writer.attrs['data_file'] = str(data_file)
        writer.attrs['label_file'] = str(label_file)
        writer.attrs['seq_num'] = seq_num
        writer.attrs['num_samples'] = num_samples
        #writer.attrs['label_names'] = names[1:]
        writer.attrs['label_names'] = names