package capture;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Clasa utilitara pentru operatii de baza cu imagini:
 * incarcare, redimensionare si conversie la tonuri de gri.
 * Toate metodele sunt statice si pot fi apelate fara a instantia clasa.
 */
public class ImageUtils {

    /**
     * Incarca o imagine de pe disc dintr-un fisier specificat.
     *
     * @param path calea completa catre fisierul imagine (PNG, JPG etc.)
     * @return un obiect BufferedImage, sau {@code null} daca incarcarea esueaza
     */
    public static BufferedImage loadImage(String path) {
        try {
            return ImageIO.read(new File(path));
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Redimensioneaza o imagine la dimensiunile dorite, pastrand raportul de aspect
     * prin scalare (SCALE_SMOOTH).
     *
     * @param src imaginea sursa
     * @param w   latimea dorita in pixeli
     * @param h   inaltimea dorita in pixeli
     * @return o noua imagine redimensionata de tip RGB
     */
    public static BufferedImage resize(BufferedImage src, int w, int h) {
        BufferedImage resized = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.drawImage(src.getScaledInstance(w, h, Image.SCALE_SMOOTH), 0, 0, null);
        g.dispose();
        return resized;
    }

    /**
     * Converteste o imagine color intr-una in tonuri de gri (grayscale).
     * Tipul de imagine rezultat este {@code BufferedImage.TYPE_BYTE_GRAY}.
     *
     * @param src imaginea sursa, colorata
     * @return o noua imagine in tonuri de gri, cu aceleasi dimensiuni
     */
    public static BufferedImage toGrayscale(BufferedImage src) {
        BufferedImage gray = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        gray.getGraphics().drawImage(src, 0, 0, null);
        return gray;
    }
}