package ui;

import features.HOG;
import capture.ImageUtils;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import java.util.Arrays;

/**
 * Fereastra de dialog pentru vizualizarea vectorului HOG al unei imagini.
 * <p>
 * Aceasta clasa este utilizata in principal din panoul de captura,
 * permitand utilizatorului sa selecteze o imagine capturata si sa
 * inspecteze descriptorul HOG corespunzator (dimensiunea 128x128).
 * <p>
 * Dialogul este modal (blocheaza fereastra principala pana la inchidere).
 */
public class HOGViewer extends JDialog {

    /**
     * Construieste dialogul care afiseaza vectorul HOG al unei imagini date.
     * <p>
     * Imaginea este redimensionata la 128x128 pixeli (dimensiunea standard
     * pentru recunoastere), convertita la tonuri de gri, apoi i se extrage
     * vectorul HOG. Rezultatul este afisat intr‑o zona de text, fara posibilitatea
     * de editare.
     *
     * @param parent    fereastra parinte (de obicei {@code MainApplication})
     * @param imageFile fisierul imagine selectat de utilizator
     */
    public HOGViewer(JFrame parent, File imageFile) {
        super(parent, "Vector HOG pentru " + imageFile.getName(), true);
        setLayout(new BorderLayout());
        try {
            BufferedImage img = ImageIO.read(imageFile);
            BufferedImage resized = ImageUtils.resize(img, 128, 128);
            double[] hog = HOG.extract(ImageUtils.toGrayscale(resized));
            JTextArea textArea = new JTextArea(20, 60);
            textArea.setText(Arrays.toString(hog));
            textArea.setEditable(false);
            add(new JScrollPane(textArea), BorderLayout.CENTER);
        } catch (Exception e) {
            add(new JLabel("Eroare la incarcare: " + e.getMessage()));
        }
        pack();
        setLocationRelativeTo(parent);
    }
}