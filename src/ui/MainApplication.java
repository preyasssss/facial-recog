package ui;

import capture.WebcamCapture;
import detector.HeadDetector;
import features.HOG;
import recognition.FaceRecognizer;
import capture.ImageUtils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.ArrayList;
import java.util.List;

/**
 * Fereastra principala a aplicatiei de recunoastere faciala.
 * <p>
 * Interfata este organizata in trei tab-uri:
 * <ul>
 *   <li><b>Captura fete</b> – permite introducerea unui pseudonim si
 *       capturarea automata a fetelor detectate de la camera web, cu
 *       un rate limiting de 200ms intre salvari pentru a asigura
 *       varietatea imaginilor si a evita duplicatele aproape identice.</li>
 *   <li><b>Antrenare</b> – ofera butoane pentru antrenarea detectorului
 *       de cap (pe baza imaginilor pozitive/negative din resources) si
 *       a clasificatoarelor de recunoastere faciala (pe baza capturilor).</li>
 *   <li><b>Recunoastere live</b> – porneste camera web si ruleaza
 *       detectia faciala pe un thread separat, iar afisarea pe EDT,
 *       asigurand un video fluid (~25fps) chiar daca detectia dureaza
 *       pana la 300ms. Fetele detectate sunt marcate cu patrat verde,
 *       iar deasupra apare pseudonimul recunoscut sau "Necunoscut".</li>
 * </ul>
 * <p>
 * La lansare, metoda statica {@link #launch()} incearca sa incarce
 * modelele existente (detectorul de cap si modelul de recunoastere),
 * apoi afiseaza fereastra.
 */
public class MainApplication extends JFrame {

    /** Panoul cu tab-uri care contine cele trei sectiuni principale. */
    private JTabbedPane tabs;

    private CapturePanel capturePanel;
    private TrainPanel trainPanel;
    private LivePanel livePanel;

    /**
     * Constructorul ferestrei principale.
     * Initializeaza cele trei panouri si le adauga in tab-uri.
     */
    public MainApplication() {
        setTitle("Recunoastere faciala cu SVM");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(900, 600);
        tabs = new JTabbedPane();
        capturePanel = new CapturePanel();
        trainPanel = new TrainPanel();
        livePanel = new LivePanel();
        tabs.addTab("Captura fete", capturePanel);
        tabs.addTab("Antrenare", trainPanel);
        tabs.addTab("Recunoastere live", livePanel);
        add(tabs);
        setLocationRelativeTo(null);
    }

    /**
     * Punctul de intrare pentru interfata grafica.
     * <p>
     * Incearca incarcarea modelelor serializate (detector de cap si
     * recunoastere faciala) si apoi afiseaza fereastra principala
     * pe firul de executie al interfetei (EDT).
     */
    public static void launch() {
        SwingUtilities.invokeLater(() -> {
            try {
                HeadDetector.load("resources/models/head_detector.svm");
                System.out.println("Model detector ok.");
            } catch (Exception e) {
                System.out.println("Model detector lipsa.");
            }
            try {
                FaceRecognizer.load("resources/models/face_recognizer.ser");
                System.out.println("Model recunoastere ok.");
            } catch (Exception e) {
                System.out.println("Model recunoastere lipsa (e ok).");
            }
            new MainApplication().setVisible(true);
        });
    }

    // ----------------- Panoul de captura -----------------
    /**
     * Panoul de captura a fetelor.
     * <p>
     * Permite utilizatorului sa introduca un pseudonim si sa porneasca
     * o sesiune de captura automata de 30 de secunde. In acest timp,
     * fiecare cadru video este procesat pentru detectia capetelor, iar
     * patch-ul este salvat in directorul corespunzator
     * ({@code resources/captured_images/<pseudonim>/}). Un rate limiting
     * de 200ms intre salvari evita duplicatele si asigura varietatea
     * imaginilor de antrenament.
     */
    class CapturePanel extends JPanel {
        private JTextField pseudonymField;
        private JLabel videoLabel, statusLabel;
        private JButton startButton, stopButton, viewHogButton;
        private WebcamCapture webcam;
        private int count = 0;
        private String currentPseudonym;
        private Timer captureTimer, stopTimer;

        // Rate limiting: timestamp-ul ultimei salvari.
        // Fara asta, la 60ms/cadru cu detectie pozitiva se salveaza ~16 poze/sec,
        // toate aproape identice. 200ms inseamna max 5 poze/sec, suficient de variat.
        private long lastSaveTime = 0;
        private static final long SAVE_INTERVAL_MS = 200;

        /**
         * Construieste panoul de captura cu toate componentele grafice.
         */
        public CapturePanel() {
            setLayout(new BorderLayout());
            JPanel top = new JPanel();
            top.add(new JLabel("Pseudonim:"));
            pseudonymField = new JTextField(10);
            top.add(pseudonymField);
            startButton = new JButton("Start captura (30 sec)");
            stopButton = new JButton("Stop");
            stopButton.setEnabled(false);
            viewHogButton = new JButton("Vizualizeaza HOG");
            viewHogButton.setEnabled(false);
            top.add(startButton);
            top.add(stopButton);
            top.add(viewHogButton);

            videoLabel = new JLabel();
            videoLabel.setPreferredSize(new Dimension(640, 480));
            videoLabel.setHorizontalAlignment(JLabel.CENTER);
            statusLabel = new JLabel("Apasati Start.");

            add(top, BorderLayout.NORTH);
            add(new JScrollPane(videoLabel), BorderLayout.CENTER);
            add(statusLabel, BorderLayout.SOUTH);

            startButton.addActionListener(e -> startCapture());
            stopButton.addActionListener(e -> stopCapture());
            viewHogButton.addActionListener(e -> showHogDialog());
        }

        /**
         * Porneste sesiunea de captura.
         * <p>
         * Verifica existenta detectorului de cap (sau il antreneaza la cerere),
         * initializeaza camera web, porneste un timer rapid (60ms) pentru
         * procesarea cadrelor si un al doilea timer (30s) care opreste automat
         * captura. Intre salvari se respecta un interval minim de 200ms pentru
         * a asigura varietatea imaginilor.
         */
        private void startCapture() {
            String name = pseudonymField.getText().trim();
            if (name.isEmpty()) {
                JOptionPane.showMessageDialog(this, "Introduceti un pseudonim!");
                return;
            }
            if (!HeadDetector.isTrained()) {
                File f = new File("resources/models/head_detector.svm");
                if (!f.exists()) {
                    int ans = JOptionPane.showConfirmDialog(this,
                            "Detectorul de cap nu exista. Antrenati acum?",
                            "Model lipsa", JOptionPane.YES_NO_OPTION);
                    if (ans == JOptionPane.YES_OPTION) {
                        HeadDetector.trainHeadDetector(
                                "resources/head_training/positive",
                                "resources/head_training/negative",
                                "resources/models/head_detector.svm");
                    } else return;
                } else {
                    HeadDetector.load("resources/models/head_detector.svm");
                }
                if (!HeadDetector.isTrained()) return;
            }

            currentPseudonym = name;
            count = 0;
            lastSaveTime = 0;

            if (webcam != null) {
                webcam.release();
                webcam = null;
            }
            try {
                webcam = new WebcamCapture(0);
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, "Eroare camera: " + ex.getMessage());
                return;
            }

            startButton.setEnabled(false);
            stopButton.setEnabled(true);
            viewHogButton.setEnabled(false);

            captureTimer = new Timer(60, ev -> {
                BufferedImage frame = webcam.grabFrame();
                if (frame == null) {
                    statusLabel.setText("Cadru NULL!");
                    return;
                }

                // afisare rapida inainte de detectie
                videoLabel.setIcon(new ImageIcon(
                    frame.getScaledInstance(640, 480, java.awt.Image.SCALE_FAST)));

                long t0 = System.currentTimeMillis();
                List<HeadDetector.Detection> detections = HeadDetector.detectHeadsFast(frame);
                long t1 = System.currentTimeMillis();

                if (!detections.isEmpty()) {
                    // deseneaza bounding box
                    Mat mat = Utils.bufferedImageToMat(frame);
                    for (HeadDetector.Detection d : detections) {
                        Imgproc.rectangle(mat, new Rect(d.x, d.y, d.width, d.height),
                                new Scalar(0, 255, 0), 2);
                    }
                    videoLabel.setIcon(new ImageIcon(Utils.matToBufferedImage(mat)));

                    // Rate limiting: salveaza maxim o poza la SAVE_INTERVAL_MS
                    long now = System.currentTimeMillis();
                    if (now - lastSaveTime >= SAVE_INTERVAL_MS) {
                        // Foloseste extractFace cu detectia deja calculata,
                        // in loc de getMaxHead(frame) care rula detectorul a doua oara
                        HeadDetector.Detection best = detections.get(0);
                        BufferedImage head = HeadDetector.extractFace(frame, best);
                        if (head != null) {
                            try {
                                File dir = new File("resources/captured_images/" + currentPseudonym);
                                if (!dir.exists()) dir.mkdirs();
                                String ts = new SimpleDateFormat("yyyyMMdd_HHmmss_SSS").format(new Date());
                                File out = new File(dir, currentPseudonym + "_" + ts + ".png");
                                ImageIO.write(head, "png", out);
                                count++;
                                lastSaveTime = now;
                            } catch (IOException ex) {
                                ex.printStackTrace();
                            }
                        }
                    }
                }
                statusLabel.setText("Capturate: " + count + " (" + (t1 - t0) + "ms)");
            });
            captureTimer.start();

            stopTimer = new Timer(60000, ev -> stopCapture());
            stopTimer.setRepeats(false);
            stopTimer.start();

            statusLabel.setText("Captureaza automat 1 minut...");
        }

        /**
         * Opreste sesiunea de captura, elibereaza camera si reseteaza controalele.
         */
        private void stopCapture() {
            if (captureTimer != null) captureTimer.stop();
            if (stopTimer != null) stopTimer.stop();
            if (webcam != null) webcam.release();
            webcam = null;
            startButton.setEnabled(true);
            stopButton.setEnabled(false);
            viewHogButton.setEnabled(true);
            videoLabel.setIcon(null);
            statusLabel.setText("Oprit. Total: " + count);
        }

        /**
         * Deschide dialogul de vizualizare a vectorului HOG pentru o imagine
         * selectata de utilizator din directorul de capturi.
         */
        private void showHogDialog() {
            JFileChooser fc = new JFileChooser("resources/captured_images");
            if (fc.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                new HOGViewer(MainApplication.this, fc.getSelectedFile()).setVisible(true);
            }
        }
    }

    // ----------------- Panoul de antrenare -----------------
    /**
     * Panoul de antrenare a modelelor.
     * <p>
     * Contine doua butoane:
     * <ul>
     *   <li><b>Antreneaza detector cap</b> – antreneaza detectorul de capete
     *       folosind imaginile din {@code resources/head_training}.</li>
     *   <li><b>Antreneaza recunoastere faciala</b> – antreneaza cate un SVM
     *       per persoana (one-vs-all) pe baza imaginilor capturate.</li>
     * </ul>
     * Progresul si rezultatele sunt afisate intr-o zona de text.
     */
    class TrainPanel extends JPanel {
        private JTextArea logArea;
        private JButton trainHeadButton, trainFaceButton;

        public TrainPanel() {
            setLayout(new BorderLayout());
            JPanel btns = new JPanel();
            trainHeadButton = new JButton("1. Antreneaza detector cap");
            trainFaceButton = new JButton("2. Antreneaza recunoastere faciala");
            btns.add(trainHeadButton);
            btns.add(trainFaceButton);
            add(btns, BorderLayout.NORTH);

            logArea = new JTextArea(10, 50);
            logArea.setEditable(false);
            add(new JScrollPane(logArea), BorderLayout.CENTER);

            trainHeadButton.addActionListener(e -> {
                trainHeadButton.setEnabled(false);
                logArea.setText("Antrenare detector cap...\n");
                new SwingWorker<Void, String>() {
                    protected Void doInBackground() {
                        try {
                            HeadDetector.trainHeadDetector(
                                    "resources/head_training/positive",
                                    "resources/head_training/negative",
                                    "resources/models/head_detector.svm");
                            HeadDetector.load("resources/models/head_detector.svm");
                            long size = new File("resources/models/head_detector.svm").length();
                            publish("Gata. Dimensiune: " + size + " octeti.");
                        } catch (Exception ex) {
                            publish("Eroare: " + ex.getMessage());
                        }
                        return null;
                    }
                    protected void process(List<String> chunks) {
                        chunks.forEach(c -> logArea.append(c + "\n"));
                    }
                    protected void done() {
                        trainHeadButton.setEnabled(true);
                    }
                }.execute();
            });

            trainFaceButton.addActionListener(e -> {
                trainFaceButton.setEnabled(false);
                logArea.setText("Antrenare recunoastere...\n");
                new SwingWorker<Void, String>() {
                    protected Void doInBackground() {
                        try {
                            FaceRecognizer.trainRecognizer(
                                    "resources/captured_images",
                                    "resources/models/face_recognizer.ser");
                            FaceRecognizer.load("resources/models/face_recognizer.ser");
                            publish("Recunoastere antrenata.");
                        } catch (Exception ex) {
                            publish("Eroare: " + ex.getMessage());
                        }
                        return null;
                    }
                    protected void process(List<String> chunks) {
                        chunks.forEach(c -> logArea.append(c + "\n"));
                    }
                    protected void done() {
                        trainFaceButton.setEnabled(true);
                    }
                }.execute();
            });
        }
    }

    // ----------------- Panoul live -----------------
    /**
     * Panoul de recunoastere live.
     * <p>
     * Arhitectura multi-thread pentru fluiditate maxima:
     * <ul>
     *   <li>Un timer Swing ({@code displayTimer}) la 40ms (~25fps) preia
     *       cadrul de la camera si il afiseaza imediat, impreuna cu ultimul
     *       bounding box cunoscut. Nu face nicio procesare grea.</li>
     *   <li>Un thread separat ({@code detectorThread}) ruleaza detectia
     *       faciala si recunoasterea. Rezultatele sunt stocate in campuri
     *       volatile, vizibile imediat pentru EDT.</li>
     *   <li>Se asigura ca doar o singura detectie ruleaza la un moment dat
     *       (flag-ul {@code detecting}), pentru a nu supraincarca CPU-ul.</li>
     * </ul>
     * Astfel, camera apare fluida (~25fps) chiar daca detectia dureaza
     * 200-300ms. Pentru fiecare fata detectata se extrage HOG-ul si se
     * interogheaza modelul de recunoastere. Daca o persoana este identificata,
     * pseudonimul apare deasupra dreptunghiului; altfel se afiseaza "Necunoscut".
     */
    class LivePanel extends JPanel {
        private JLabel videoLabel;
        private JButton startButton, stopButton;
        private Timer displayTimer;
        private WebcamCapture webcam;

        // Starea partajata intre EDT si thread-ul de detectie.
        // volatile garanteaza ca EDT vede mereu ultima valoare scrisa de detector.
        private volatile BufferedImage lastFrame = null;
        private volatile List<HeadDetector.Detection> lastDetections = new ArrayList<>();
        private volatile String lastLabel = "";
        private volatile boolean detecting = false;

        // Thread dedicat pentru detectie + recunoastere (nu blocheaza EDT)
        private java.util.concurrent.ExecutorService detectorThread =
            java.util.concurrent.Executors.newSingleThreadExecutor();

        public LivePanel() {
            setLayout(new BorderLayout());
            JPanel top = new JPanel();
            startButton = new JButton("Start recunoastere live");
            stopButton = new JButton("Stop");
            stopButton.setEnabled(false);
            top.add(startButton);
            top.add(stopButton);
            add(top, BorderLayout.NORTH);

            videoLabel = new JLabel();
            videoLabel.setPreferredSize(new Dimension(640, 480));
            add(videoLabel, BorderLayout.CENTER);

            startButton.addActionListener(e -> startLive());
            stopButton.addActionListener(e -> stopLive());
        }

        /**
         * Porneste camera web si thread-urile pentru recunoastere live.
         */
        private void startLive() {
            if (!HeadDetector.isTrained()) {
                JOptionPane.showMessageDialog(this, "Detectorul de cap nu este antrenat.");
                return;
            }
            if (webcam != null) { webcam.release(); webcam = null; }
            try {
                webcam = new WebcamCapture(0);
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, "Eroare camera: " + ex.getMessage());
                return;
            }

            lastDetections = new ArrayList<>();
            lastLabel = "";
            detecting = false;

            startButton.setEnabled(false);
            stopButton.setEnabled(true);

            // Timer de afisare — ruleaza la 40ms (~25fps) si nu face nimic greu.
            // Preia cadrul, deseneaza ultimul bounding box cunoscut si
            // lanseaza detectia in background daca nu e una in curs.
            displayTimer = new Timer(40, ev -> {
                BufferedImage frame;
                try {
                    frame = webcam.grabFrame();
                } catch (Exception ex) { return; }
                if (frame == null) return;
                lastFrame = frame;

                // Deseneaza mereu ultimul bounding box, indiferent de viteza detectiei.
                // Astfel camera apare fluida chiar daca detectia dureaza 300ms.
                List<HeadDetector.Detection> dets = lastDetections;
                if (!dets.isEmpty()) {
                    Mat mat = Utils.bufferedImageToMat(frame);
                    for (HeadDetector.Detection d : dets) {
                        Rect r = new Rect(d.x, d.y, d.width, d.height);
                        Imgproc.rectangle(mat, r, new Scalar(0, 255, 0), 2);
                        Imgproc.putText(mat, lastLabel,
                                new org.opencv.core.Point(r.x, r.y - 5),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 0.6,
                                new Scalar(0, 255, 0), 2);
                    }
                    videoLabel.setIcon(new ImageIcon(Utils.matToBufferedImage(mat)));
                } else {
                    videoLabel.setIcon(new ImageIcon(
                        frame.getScaledInstance(640, 480, java.awt.Image.SCALE_FAST)));
                }

                // Lanseaza o noua detectie async doar daca precedenta s-a terminat
                if (!detecting) {
                    detecting = true;
                    final BufferedImage snapshot = frame;
                    detectorThread.submit(() -> {
                        try {
                            List<HeadDetector.Detection> found =
                                HeadDetector.detectHeadsFast(snapshot);
                            lastDetections = found;

                            if (!found.isEmpty()) {
                                HeadDetector.Detection best = found.get(0);
                                BufferedImage face = HeadDetector.extractFace(snapshot, best);
                                String name = "Necunoscut";
                                if (face != null && FaceRecognizer.persons != null
                                        && !FaceRecognizer.persons.isEmpty()) {
                                    double[] hog = HOG.extract(
                                        ImageUtils.resize(face, 128, 128));
                                    String id = FaceRecognizer.identify(
                                        FaceRecognizer.normalizeL2(hog));
                                    if (id != null) name = id;
                                }
                                lastLabel = name;
                            } else {
                                lastLabel = "";
                            }
                        } catch (Exception ex) {
                            System.err.println("Eroare detector: " + ex.getMessage());
                        } finally {
                            detecting = false; // elibereaza slot pentru urmatoarea detectie
                        }
                    });
                }
            });
            displayTimer.start();
        }

        /**
         * Opreste camera web si toate thread-urile, eliberand resursele.
         */
        private void stopLive() {
            if (displayTimer != null) displayTimer.stop();
            if (webcam != null) webcam.release();
            webcam = null;
            lastFrame = null;
            lastDetections = new ArrayList<>();
            startButton.setEnabled(true);
            stopButton.setEnabled(false);
            videoLabel.setIcon(null);
        }
    }
}