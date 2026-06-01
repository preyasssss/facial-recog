package ui;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

/**
 * Clasa utilitara pentru conversii intre obiectele de tip {@link Mat} (OpenCV)
 * si {@link BufferedImage} (Java standard).
 * <p>
 * Aceste conversii sunt necesare deoarece OpenCV este utilizat strict pentru
 * captura video si desenarea dreptunghiurilor/textului, iar restul
 * procesarii se face pe imagini Java standard.
 */
public class Utils {

    /**
     * Converteste o imagine Java {@link BufferedImage} intr-o matrice OpenCV
     * de tip 8-bit, 3 canale (BGR).
     *
     * @param bi imaginea de intrare (trebuie sa fie de tip 3BYTE_BGR)
     * @return matricea OpenCV corespunzatoare
     */
    public static Mat bufferedImageToMat(BufferedImage bi) {
        Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
        byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
        mat.put(0, 0, data);
        return mat;
    }

    /**
     * Converteste o matrice OpenCV (BGR, 3 canale) intr-o imagine Java
     * {@link BufferedImage} compatibila cu componentele Swing.
     *
     * @param mat matricea OpenCV de convertit
     * @return imaginea Java, de tip 3BYTE_BGR
     */
    public static BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_3BYTE_BGR;
        BufferedImage img = new BufferedImage(mat.cols(), mat.rows(), type);
        byte[] data = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        mat.get(0, 0, data);
        return img;
    }
}