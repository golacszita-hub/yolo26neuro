## Project Goal

This project investigates whether YOLO can detect and classify
neuronal spike waveforms represented as 2D images.
Currently the immediate goal is to have YOLO26n model overfit a tiny fixed dataset of approximately 100 mean waveform images extracted from a neural nwb file. 
A successful overfit test is a prerequisite for further
development.

The current test contains:

100 neuronal units
81 waveform time samples
128 electrode channels
1 mean waveform per unit
100 images generated from these mean waveforms
1 class per neuronal unit

The corresponding image representation is approximately:

100 images
×
128 × 128 pixels

