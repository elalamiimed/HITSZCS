package edu.hitsz.music;

import javax.sound.sampled.*;
import javax.sound.sampled.DataLine.Info;
import java.io.*;

/**
 * A thread class for playing audio files using Java Sound API.
 * Supports optional looping and on-demand stopping.
 */

public class MusicThread extends Thread {


    // Flag indicating whether the audio should loop continuously
    private boolean isRepeat = false;

    // Flag to signal that playback should stop immediately
    private boolean isStop = false;

    // Path to the audio file to be played
    private final String filename;

    // The audio format of the loaded fil
    private AudioFormat audioFormat;

    // Raw audio data as a byte array, loaded from the file
    private byte[] samples;

    /**
     * Constructor: initializes the music player with a given file and repeat setting.
     *
     * @param filename  Path to the audio file (e.g., "music.wav")
     * @param isRepeat  Whether to loop the audio after it finishes
     */
    public MusicThread(String filename, boolean isRepeat)
    {
        this.filename=filename;
        this.isRepeat = isRepeat;
        reverseMusic();
    }

    /**
     * Loads the audio file and extracts its format and raw sample data.
     * This method reads the file once at initialization to avoid repeated I/O during playback.
     */
    public void reverseMusic() {
        try {
            // Obtain an AudioInputStream from the specified file
            AudioInputStream stream = AudioSystem.getAudioInputStream(new File(filename));
            //Use the AudioFormat to obtain the format of the AudioInputStream
            audioFormat = stream.getFormat();
            // Extract all audio samples into a byte array
            samples = getSamples(stream);
        } catch (UnsupportedAudioFileException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Reads all audio samples from the given AudioInputStream into a byte array.
     *
     * @param stream  The input stream containing the audio data
     * @return        A byte array containing the complete audio samples
     */
    public byte[] getSamples(AudioInputStream stream) {
        // Calculate total number of bytes: frames × bytes per frame
        int size = (int) (stream.getFrameLength() * audioFormat.getFrameSize());
        byte[] samples = new byte[size];
        DataInputStream dataInputStream = new DataInputStream(stream);
        try {
            // Read the entire audio data into the byte array
            dataInputStream.readFully(samples);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return samples;
    }

    /**
     * Plays audio from the given InputStream using a SourceDataLine.
     *
     * @param source  InputStream containing audio data to play
     */
    public void play(InputStream source) {
        // Buffer size: one second of audio data (frame size × sample rate)
        int size = (int) (audioFormat.getFrameSize() * audioFormat.getSampleRate());
        byte[] buffer = new byte[size];

        // SourceDataLine: outputs audio data to the mixer (speakers)
        SourceDataLine dataLine = null;
        // Describe the desired audio line: SourceDataLine with the loaded format
        Info info = new Info(SourceDataLine.class, audioFormat);

        try {
            // Obtain and open the audio output line
            dataLine = (SourceDataLine) AudioSystem.getLine(info);
            dataLine.open(audioFormat, size);
        } catch (LineUnavailableException e) {
            e.printStackTrace();
        }
        // Start audio playback
        dataLine.start();
        try {
            int numBytesRead = 0;
            while (numBytesRead != -1) {
                // Check if external stop signal has been received
                if (this.isStop) {
                    isRepeat = false; // Ensure loop stops
                    break;
                }

                //Read up to the specified maximum number of bytes from the audio stream into the buffer
                numBytesRead =
                        source.read(buffer, 0, buffer.length);
                //Write data to the mixer via this source data line
                if (numBytesRead != -1) {
                    dataLine.write(buffer, 0, numBytesRead);
                }
            }

        } catch (IOException ex) {
            ex.printStackTrace();
        }

        dataLine.drain();
        dataLine.close();

    }

    /**
     * Main thread execution method.
     * Plays the audio repeatedly if isRepeat is true.
     */
    @Override
    public void run() {
        do {
            // Create a new input stream from the pre-loaded samples
            InputStream stream = new ByteArrayInputStream(samples);
            play(stream);
        } while (isRepeat);
    }

    /**
     * Signals the thread to stop playback immediately.
     */
    public void stopPlay() {
        this.isStop = true;
    }
}


