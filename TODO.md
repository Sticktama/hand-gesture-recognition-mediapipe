# LSTM Data Collection Implementation

## Tasks

- [x] Modify `data_collect_lstm.py` to collect data from PNG files or videos
- [x] Implement the EXACT same preprocessing algorithms as in app.py
- [x] Create a function to process video files frame by frame
- [x] Implement sequence collection for LSTM (16 timesteps)
- [x] Save data in the correct format for LSTM training
- [ ] Test the implementation with sample videos/images

## Implementation Details

1. **Data Collection Methods**:
   - Process video files with `process_video()` function
   - Process image sequences with `process_image_sequence()` function
   - Both methods track hand landmarks across frames

2. **LSTM Data Format**:
   - Collects sequences of 16 timesteps (configurable via TIME_STEPS constant)
   - Each timestep contains X,Y coordinates (2 dimensions)
   - Tracks index finger tip position (landmark[8])
   - Normalizes coordinates relative to the first point in the sequence

3. **Usage**:
   - Place videos (.mp4, .avi, .mov) or image sequences in class directories
   - Run `python data_collect_lstm.py` to process all classes
   - Output saved to "model/point_history_classifier/point_history.csv"

4. **Next Steps**:
   - Test with sample videos/images
   - Run LSTM training using point_history_classification.ipynb (set use_lstm = True)
   - Evaluate model performance
