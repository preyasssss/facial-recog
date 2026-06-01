package detector;

import features.HOG;
import capture.ImageUtils;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;

/**
 * Clasificator de capete (fete) bazat pe similaritatea cosinus
 * fata de un vector HOG mediu pozitiv normalizat L2.
 * <p>
 * Antrenamentul:
 * <ol>
 *   <li>Incarca imaginile pozitive (fete) si extrage descriptorii HOG.</li>
 *   <li>Calculeaza vectorul HOG mediu al exemplelor pozitive si il
 *       normalizeaza la norma unitara (L2).</li>
 *   <li>Determina similaritatea fiecarui exemplu pozitiv fata de vectorul
 *       mediu si stabileste pragul de decizie ca fiind percentila 10% --
 *       un echilibru intre sensibilitate si precizie. (La percentila 5%
 *       apar prea multi falsi pozitivi, la 20% detectia devine prea rara.)</li>
 *   <li>Salveaza vectorul mediu si pragul prin serializare.</li>
 * </ol>
 * <p>
 * Detectia:
 * <ul>
 *   <li>Parcurge imaginea la mai multe scari (piramida), cu o fereastra
 *       glisanta de 64x64 pixeli si un pas de 16 pixeli (de 4 ori mai
 *       putine ferestre decat pasul 8, diferenta vizuala minima).</li>
 *   <li>Aplica un pre-filtru rapid care verifica intr-o singura trecere
 *       variatia luminantei (elimina zonele uniforme) si prezenta culorii
 *       pielii in spatiul YCbCr (cel putin 30% pixeli de piele).</li>
 *   <li>Extrage HOG pentru ferestrele care trec de pre-filtru si calculeaza
 *       similaritatea cosinus cu vectorul mediu.</li>
 *   <li>Pastreaza detecțiile cu scor peste prag si le filtreaza prin
 *       suprimare non-maxima (IoU 0.6), returnand cel mult o detectie
 *       (cea mai buna).</li>
 * </ul>
 * <p>
 * Pentru afisare live fluida, metoda {@link #detectHeadsFast} redimensioneaza
 * imaginea la 320px latime inainte de detectie si remapeaza coordonatele
 * la dimensiunea originala.
 * <p>
 * Metoda {@link #extractFace} expandeaza fereastra de detectie cu 50% si
 * o muta usor in jos (25% din inaltimea originala), pentru a include intreg
 * capul pana la barbie, nu doar fruntea/crestetul unde se declanseaza
 * de obicei fereastra de 64px.
 */
public class HeadDetector {

    /** Vectorul HOG mediu al exemplelor pozitive, normalizat L2. */
    private static double[] meanPositiveVector;

    /** Pragul de similaritate cosinus peste care o fereastra este acceptata. */
    private static double threshold = 0.0;

    /**
     * Verifica daca detectorul a fost antrenat si modelul este incarcat.
     * @return {@code true} daca vectorul mediu exista
     */
    public static boolean isTrained() {
        return meanPositiveVector != null;
    }

    /**
     * Antreneaza detectorul de capete folosind imaginile pozitive si negative
     * din directoarele specificate.
     * <p>
     * Procesul:
     * <ol>
     *   <li>Incarca imaginile pozitive, le redimensioneaza la 64x64 si
     *       extrage descriptorii HOG.</li>
     *   <li>Calculeaza vectorul HOG mediu al exemplelor pozitive si il
     *       normalizeaza L2.</li>
     *   <li>Determina similaritatea fiecarui exemplu pozitiv fata de vectorul
     *       mediu si stabileste pragul ca fiind percentila 10%, asigurand
     *       un echilibru intre sensibilitate si precizie.</li>
     *   <li>Salveaza modelul (vectorul mediu si pragul) prin serializare.</li>
     * </ol>
     *
     * @param posDir    directorul cu imagini pozitive (fete, 64x64 .png/.jpg)
     * @param negDir    directorul cu imagini negative (fundal, 64x64 .png/.jpg)
     * @param modelPath calea fisierului unde se salveaza modelul (.svm)
     */
    public static void trainHeadDetector(String posDir, String negDir, String modelPath) {
        List<double[]> posFeatures = new ArrayList<>();
        File[] posFiles = new File(posDir).listFiles((d, n) -> n.endsWith(".png") || n.endsWith(".jpg"));
        if (posFiles == null || posFiles.length == 0) {
            System.out.println("EROARE: niciun fisier pozitiv gasit in " + posDir);
            return;
        }
        for (File f : posFiles) {
            BufferedImage img = ImageUtils.loadImage(f.getAbsolutePath());
            if (img == null) continue;
            BufferedImage resized = ImageUtils.resize(img, 64, 64);
            double[] hog = HOG.extract(resized);
            posFeatures.add(hog);
        }
        System.out.println("Pozitive incarcate: " + posFeatures.size());

        int len = posFeatures.get(0).length;
        meanPositiveVector = new double[len];
        for (double[] vec : posFeatures) {
            for (int i = 0; i < len; i++) meanPositiveVector[i] += vec[i];
        }
        for (int i = 0; i < len; i++) meanPositiveVector[i] /= posFeatures.size();

        // normalizare L2
        double norm = 0.0;
        for (double v : meanPositiveVector) norm += v * v;
        norm = Math.sqrt(norm);
        if (norm < 1e-9) {
            System.out.println("Vector mediu nul!");
            return;
        }
        for (int i = 0; i < len; i++) meanPositiveVector[i] /= norm;

        double[] sims = new double[posFeatures.size()];
        for (int i = 0; i < sims.length; i++) {
            sims[i] = cosineSimilarity(posFeatures.get(i), meanPositiveVector);
        }
        Arrays.sort(sims);

        // Percentila 10: echilibru intre sensibilitate si precizie.
        // La 5% erau prea multi falsi pozitivi, la 20% detectia era prea rara.
        threshold = sims[(int)(0.10 * sims.length)];
        System.out.println("Prag automat (10% percentila): " + threshold);

        save(modelPath);
        System.out.println("Detector salvat.");
    }

    /**
     * Calculeaza similaritatea cosinus intre doi vectori.
     * @param a primul vector
     * @param b al doilea vector
     * @return o valoare intre -1 si 1 (1 = identici, 0 = ortogonali)
     */
    private static double cosineSimilarity(double[] a, double[] b) {
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot  += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA < 1e-9 || normB < 1e-9) return 0.0;
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Incarca un model salvat anterior (vectorul mediu si pragul).
     * @param path calea fisierului serializat
     */
    public static void load(String path) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            meanPositiveVector = (double[]) ois.readObject();
            threshold = ois.readDouble();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void save(String path) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(meanPositiveVector);
            oos.writeDouble(threshold);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Pre-filtru combinat: verifica variatia luminantei si prezenta culorii
     * pielii intr-o singura parcurgere a pixelilor ferestrei.
     * <p>
     * Inlocuieste metodele separate {@code hasEnoughVariance} si
     * {@code hasSkinPixels} (doua loop-uri separate), pentru eficienta.
     * <p>
     * Pielea este detectata in spatiul YCbCr cu praguri standard:
     * {@code Y > 80, Cb in (77,127), Cr in (133,173)}.
     * <p>
     * Intoarce {@code true} daca varianta pixelilor depaseste 100
     * si cel putin 30% din pixeli sunt de culoarea pielii.
     *
     * @param window fereastra de 64x64 pixeli de analizat
     * @return {@code true} daca fereastra trece ambele filtre
     */
    private static boolean passesPrefilter(BufferedImage window) {
        int total = 0, skin = 0;
        double sum = 0, sumSq = 0;
        int w = window.getWidth(), h = window.getHeight();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = window.getRGB(x, y);
                int r  = (rgb >> 16) & 0xff;
                int g  = (rgb >> 8)  & 0xff;
                int bv =  rgb        & 0xff;

                // luminanta pentru variance
                int gray = (r + g + bv) / 3;
                sum   += gray;
                sumSq += (double) gray * gray;

                // YCbCr pentru skin
                double Y  =  0.299    * r + 0.587    * g + 0.114    * bv;
                double Cb = -0.168736 * r - 0.331264 * g + 0.5      * bv + 128;
                double Cr =  0.5      * r - 0.418688 * g - 0.081312 * bv + 128;
                total++;
                if (Y > 80 && Cb > 77 && Cb < 127 && Cr > 133 && Cr < 173) skin++;
            }
        }
        double mean     = sum / total;
        double variance = (sumSq / total) - (mean * mean);
        return variance > 100 && (double) skin / total > 0.3;
    }

    /**
     * Detecteaza capete intr-o imagine la scara completa.
     * <p>
     * Algoritmul foloseste o piramida de imagini (scara initiala 1.0,
     * multiplicata cu 0.7 la fiecare nivel, pana cand dimensiunea ferestrei
     * scade sub 80px). La fiecare nivel, o fereastra glisanta de 64x64 pixeli
     * cu pasul 16 este evaluata de pre-filtru si, daca trece, i se extrage
     * HOG si i se calculeaza similaritatea cosinus cu vectorul mediu.
     * <p>
     * Detecțiile cu scor peste prag sunt pastrate si filtrate prin suprimare
     * non-maxima (IoU 0.6). Daca dupa NMS raman mai multe, se pastreaza doar
     * cea cu scorul maxim.
     *
     * @param image imaginea in care se cauta capete
     * @return lista de detecții (de obicei un singur element)
     */
    public static List<Detection> detectHeads(BufferedImage image) {
        List<Detection> detections = new ArrayList<>();
        int width  = image.getWidth();
        int height = image.getHeight();
        double scale  = 1.0;
        BufferedImage scaled = image;

        while (scaled.getWidth() >= 64 && scaled.getHeight() >= 64) {
            int step = 16; // era 8 — 4x mai putine ferestre
            for (int y = 0; y <= scaled.getHeight() - 64; y += step) {
                for (int x = 0; x <= scaled.getWidth() - 64; x += step) {
                    BufferedImage sub = scaled.getSubimage(x, y, 64, 64);
                    if (!passesPrefilter(sub)) continue;
                    double[] hog   = HOG.extract(sub);
                    double   score = cosineSimilarity(hog, meanPositiveVector);
                    if (score > threshold) {
                        int origX    = (int)(x  / scale);
                        int origY    = (int)(y  / scale);
                        int origSize = (int)(64 / scale);
                        if (origSize >= 50) {
                            detections.add(new Detection(origX, origY, origSize, origSize, score));
                        }
                    }
                }
            }
            scale *= 0.7;
            int newW = (int)(width  * scale);
            int newH = (int)(height * scale);
            if (newW < 80 || newH < 80) break;
            scaled = ImageUtils.resize(image, newW, newH);
        }

        List<Detection> afterNMS = NonMaxSuppression.suppress(detections, 0.6);
        if (afterNMS.size() > 1) {
            afterNMS.sort((a, b) -> Double.compare(b.score, a.score));
            return afterNMS.subList(0, 1);
        }
        return afterNMS;
    }

    /**
     * Versiune rapida a detectiei, pentru afisare live fluida.
     * <p>
     * Redimensioneaza imaginea la 320px latime, aplica
     * {@link #detectHeads(BufferedImage)} si remapeaza coordonatele
     * la dimensiunea originala.
     *
     * @param image imaginea originala (de obicei 640x480)
     * @return detecțiile scalate corect pentru imaginea originala
     */
    public static List<Detection> detectHeadsFast(BufferedImage image) {
        int    targetWidth  = 320;
        double ratio        = (double) targetWidth / image.getWidth();
        int    targetHeight = (int)(image.getHeight() * ratio);
        BufferedImage small = ImageUtils.resize(image, targetWidth, targetHeight);
        List<Detection> detections = detectHeads(small);
        for (Detection d : detections) {
            d.x      = (int)(d.x      / ratio);
            d.y      = (int)(d.y      / ratio);
            d.width  = (int)(d.width  / ratio);
            d.height = (int)(d.height / ratio);
        }
        return detections;
    }

    /**
     * Extrage si redimensioneaza fata din cadrul dat, folosind o detectie
     * deja calculata. Evita o a doua rulare a detectorului.
     * <p>
     * Fereastra de detectie de 64px se declanseaza adesea pe frunte/crestet.
     * Expandam bbox-ul cu 50% si il mutam usor in jos (25% din inaltime)
     * ca sa includem tot capul pana la barbie, nu doar parul.
     *
     * @param frame     imaginea originala de la camera
     * @param detection detectia returnata de {@link #detectHeadsFast}
     * @return patch 128x128 sau {@code null} daca coordonatele ies din imagine
     */
    public static BufferedImage extractFace(BufferedImage frame, Detection detection) {
        // centrul ferestrei detectate
        int cx = detection.x + detection.width  / 2;
        int cy = detection.y + detection.height / 2;

        // marim cu 50% si coboram centrul cu 25% din dimensiunea originala
        // (detectia loveste de obicei fruntea — facem loc barbiei)
        int expanded = (int)(detection.width * 1.5);
        int shiftDown = (int)(detection.height * 0.25);
        cy += shiftDown;

        int x = Math.max(0, cx - expanded / 2);
        int y = Math.max(0, cy - expanded / 2);
        int w = Math.min(expanded, frame.getWidth()  - x);
        int h = Math.min(expanded, frame.getHeight() - y);
        if (w <= 0 || h <= 0) return null;
        return ImageUtils.resize(frame.getSubimage(x, y, w, h), 128, 128);
    }

    /**
     * Metoda pastrata pentru compatibilitate, dar nu mai este folosita
     * in noul cod (ruleaza detectia a doua oara inutil).
     *
     * @param image imaginea sursa
     * @return patch-ul cu fata, 128x128, sau {@code null}
     * @deprecated folositi {@link #extractFace} impreuna cu o detectie existenta
     */
    @Deprecated
    public static BufferedImage getMaxHead(BufferedImage image) {
        List<Detection> heads = detectHeads(image);
        if (heads.isEmpty()) return null;
        heads.sort((a, b) -> Double.compare(b.score, a.score));
        return extractFace(image, heads.get(0));
    }

    /**
     * Clasa interna care reprezinta o detectie (dreptunghi + scor).
     */
    public static class Detection {
        /** Coordonata stanga-sus pe OX. */
        public int x;
        /** Coordonata stanga-sus pe OY. */
        public int y;
        /** Latimea dreptunghiului. */
        public int width;
        /** Inaltimea dreptunghiului. */
        public int height;
        /** Scorul de incredere (similaritate cosinus). */
        public double score;

        /**
         * Creeaza o noua detectie.
         * @param x coordonata X
         * @param y coordonata Y
         * @param w latime
         * @param h inaltime
         * @param s scor
         */
        public Detection(int x, int y, int w, int h, double s) {
            this.x = x; this.y = y; this.width = w; this.height = h; this.score = s;
        }
    }
}