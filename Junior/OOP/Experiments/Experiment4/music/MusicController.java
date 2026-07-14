package edu.hitsz.music;
/**
 * Audio controller class for managing background music and sound effects in a game.
 *
 * Guidelines:
 * - Background music (BGM) and boss music should loop continuously and must be
 *   controlled via dedicated thread references so they can be stopped later.
 * - One-time sound effects (e.g., explosion, item pickup) should be played once
 *   and do not need to be stored as instance variables.
 *
 * Tasks for students:
 * 1. Refer to the implemented methods startBgmMusic() and stopBgmMusic().
 * 2. Complete all methods marked with "TODO".
 * 3. Remember:
 *    - Looping sounds: create and store a MusicThread (like bgmThread).
 *    - One-shot sounds: create a new MusicThread(..., false) and call .start() immediately.
 */
public class MusicController {

    // File paths for audio resources
    public static String bgmMusic = "src/videos/bgm.wav";
    public static String bossMusic = "src/videos/bgm_boss.wav";


    // Keep references to looping music threads so we can stop them later
    private MusicThread bgmThread;
    private MusicThread bossThread;

    /**
     * Constructor: initializes the background music thread (set to loop).
     */
    public MusicController() {
        bgmThread = new MusicThread(bgmMusic, true);
    }

    /**
     * Starts playing the background music (loops indefinitely).
     */
    public void startBgmMusic() {
        bgmThread.start();
    }

    /**
     * Stops the background music.
     */
    public void stopBgmMusic() {
        bgmThread.stopPlay();
    }

    /**
     * Starts playing the boss music (should loop).
     *
     * TODO: Implement this method by creating and starting a looping MusicThread for bossMusic.
     *       Store it in the bossThread field so it can be stopped later.
     */
    public void startBossMusic() {
        // TODO
    }

    /**
     * Stops the boss music.
     *
     * TODO: Stop the bossThread if it is not null.
     */
    public void stopBossMusic() {
        // TODO
    }

    /**
     * Plays the bullet hit sound effect (one-time playback).
     *
     * TODO: Create a new non-looping MusicThread for bulletHitMusic and start it immediately.
     */
    public void startBulletHitMusic() {
        // TODO
    }

    /**
     * Plays the supply pickup sound effect (one-time playback).
     *
     * TODO: Create a new non-looping MusicThread for supplyMusic and start it immediately.
     */
    public void startSupplyMusic() {
        // TODO
    }


    /**
     * Plays the game over sound effect (one-time playback).
     *
     * TODO: Create a new non-looping MusicThread for gameOverMusic and start it immediately.
     */
    public void startGameOverMusic() {
        // TODO
    }
}