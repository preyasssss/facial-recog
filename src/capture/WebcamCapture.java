package capture;

import org.opencv.core.*;
import org.opencv.videoio.VideoCapture;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

/**
 * Clasa responsabila cu interactiunea directa cu camera web,
 * folosind biblioteca OpenCV (singura utilizare permisa a OpenCV
 * in afara desenarii dreptunghiurilor si textului).
 * <p>
 * Ofera metode pentru pornirea camerei, obtinerea cadrelor
 * sub forma de {@link BufferedImage} si eliberarea resurselor.
 */
public class WebcamCapture {

    /* Bloc static care incarca biblioteca nativa OpenCV o singura data,
       la prima referire a clasei. */
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    /** Obiectul OpenCV care reprezinta legatura cu camera web. */
    private VideoCapture cap;

    /** Latimea ferestrei video capturate (implicit 640). */
    private int width;

    /** Inaltimea ferestrei video capturate (implicit 480). */
    private int height;

    /**
     * Initializeaza camera web cu indexul specificat.
     * Configureaza rezolutia standard de 640x480 pixeli.
     *
     * @param cameraIndex indexul camerei (0 = camera principala)
     */
    public WebcamCapture(int cameraIndex) {
        cap = new VideoCapture(cameraIndex);
        cap.set(org.opencv.videoio.Videoio.CAP_PROP_FRAME_WIDTH, 640);
        cap.set(org.opencv.videoio.Videoio.CAP_PROP_FRAME_HEIGHT, 480);
        width  = (int) cap.get(org.opencv.videoio.Videoio.CAP_PROP_FRAME_WIDTH);
        height = (int) cap.get(org.opencv.videoio.Videoio.CAP_PROP_FRAME_HEIGHT);
    }

    /**
     * Preia un cadru curent de la camera web si il converteste
     * dintr-un obiect OpenCV {@code Mat} intr-un {@code BufferedImage}
     * compatibil cu Java Swing.
     *
     * @return imaginea capturata, sau {@code null} daca citirea esueaza
     */
    public BufferedImage grabFrame() {
        Mat mat = new Mat();
        if (cap.read(mat)) {
            return matToBufferedImage(mat);
        }
        return null;
    }

    /**
     * Elibereaza resursele camerei web.
     * Trebuie apelata cand captura nu mai este necesara.
     */
    public void release() {
        cap.release();
    }

    /**
     * Converteste o matrice OpenCV ({@code Mat}) intr-un obiect
     * {@code BufferedImage} Java.
     * <p>
     * Se foloseste tipul BGR (3 canale) pentru imaginile color si
     * tipul BYTE_GRAY pentru imaginile cu un singur canal.
     *
     * @param mat matricea OpenCV de convertit
     * @return imaginea Java corespunzatoare
     */
    private BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_3BYTE_BGR;
        if (mat.channels() == 1) {
            type = BufferedImage.TYPE_BYTE_GRAY;
        }
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        mat.get(0, 0, data);
        return image;
    }
}